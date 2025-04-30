from agreement.metrics import krippendorffs_alpha
import functools
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import pingouin as pg
import scipy.stats as stats
import seaborn as sns

from statsmodels.stats import inter_rater as irr
from sklearn.preprocessing import MinMaxScaler

import utils
import visualizations

'''

DATA CLEANING / PREPARATION

'''
def get_stats_df(df, stats, axis_name):
    '''
    Given DF of numbers, return the statistics requested
    '''
    # Convert data to float
    df = df.map(lambda x: pd.to_numeric(x, errors='coerce'))

    stats_df = {}
    for stat in stats:
        if stat == 'mean':
            calculated = df.mean(axis=0) # automatically skips NaN
        elif stat == 'std':
            calculated = df.std(axis=0) # automatically skips NaN
        elif stat == 'median':
            calculated = df.median(axis=0) # automatically skips NaN
        else:
            raise ValueError("Stat {} not supported".format(stat))
        stats_df[stat] = calculated

    stats_df = pd.DataFrame(stats_df)
    stats_df = stats_df.rename_axis(axis_name).reset_index()
    return stats_df

'''
Functions for timing analyses
'''

def time_analysis(df,
                  mapping,
                  metadata_mapping,
                  units='minutes',
                  condition='all',
                  save_dir=None,
                  overwrite=False,
                  debug=True):
    '''
    Given a DF of raw data, output the time analysis DF
    '''
    # If file exists, return it
    if save_dir is not None:
        save_path = os.path.join(save_dir, 'time_analysis_{}_{}.csv'.format(condition, units))
        if os.path.exists(save_path) and not overwrite:
            utils.informal_log("Time analysis exists at {}".format(save_path))
            return utils.read_file(save_path)

    # Drop rows that are not responses
    df = df[df['ResponseId'].str.startswith('R_')]

    time_df = {}
    df_columns = df.columns
    n_rows = None

    # Keep metadata
    for mdata_col_name, mdata_new_col_name in metadata_mapping.items():
        column = df[mdata_col_name]

        if mdata_col_name == 'Duration (in seconds)' and units == 'minutes':
            column = pd.to_numeric(df[mdata_col_name], errors='coerce')
            column /= 60
        time_df[mdata_new_col_name] = column

        if n_rows is None:
            n_rows = len(column)
        else:
            assert len(column) == n_rows, "Number of rows {} did not match expected ({})".format(len(column), n_rows)

    section_columns = {}
    for section, col_mapping in mapping.items():
        # Keep track of new column names for all columns in this section
        cur_section_columns = []
        for q_num, name in col_mapping.items():
            # Get column for time on each page & make new column name
            col_name = '{}_Page Submit'.format(q_num)
            new_col_name = '{}_{}'.format(section, name)

            # If item is survey, need to sum up times across each page
            if 'survey_time' in name:
                # Select subset of columns that belong to timing of survey
                survey_time_df = df[df.columns[df.columns.str.contains(pat=col_name)]]  # N x 40


                assert survey_time_df.isnull().sum().sum() == 0
                # Convert data type from str -> float
                survey_time_df = survey_time_df.astype(float)
                if save_dir is not None:
                    survey_time_save_path = os.path.join(save_dir, 'anthro_survey_time.csv')

                    # Add condition & PID; rename columns
                    save_survey_time_df = pd.concat([survey_time_df, df['CONDITION'], df['PROLIFIC_PID']], axis=1)
                    save_survey_time_df.rename({'CONDITION': 'condition', 'PROLIFIC_PID': 'participant_id'}, axis=1)
                    utils.write_file(save_survey_time_df, survey_time_save_path, overwrite=overwrite)
                # Sum across columns
                column = survey_time_df.sum(axis=1)

                # Also add median, min time on each survey page
                median_time = survey_time_df.median(axis=1)
                min_time = survey_time_df.min(axis=1)
                max_time = survey_time_df.max(axis=1)
                if units == 'minutes':
                    median_time /= 60
                    min_time /= 60
                    max_time /= 60
                time_df['median_time_per_survey_page'] = median_time
                time_df['min_time_per_survey_page'] = min_time
                time_df['max_time_per_survey_page'] = max_time


            # Otherwise each item is from a single column
            else:
                if col_name not in df_columns:
                    if (condition == 'baseline' or condition == 'all') and 'video' in section:
                        continue
                    utils.informal_log("Column {} ({}) not in DF columns in {} condition".format(
                        col_name, new_col_name, condition))
                    continue

                # Convert to floats & add as new column in new DF
                column = pd.to_numeric(df[col_name], errors='coerce')

            if n_rows is None:
                n_rows = len(column)
            else:
                assert len(column) == n_rows, "Number of rows {} did not match expected ({})".format(len(column), n_rows)

            if units == 'minutes':
                column /= 60

            time_df[new_col_name] = column
            cur_section_columns.append(new_col_name)

        section_columns[section] = cur_section_columns

    time_df = pd.DataFrame(time_df)

    # Create columns for mechanistic, functional, and intentional videos
    modes = ['video_mech', 'video_func', 'video_intent']
    for mode in modes:
        mode_df = time_df.filter(regex=mode)
        time_df[mode] = mode_df.sum(axis=1, min_count=1)

    # For each section, sum up the total time in each section & add as a column
    for section, columns in section_columns.items():
        section_times = time_df[columns].sum(axis=1)
        time_df[section] = section_times

    if save_dir is not None:
        utils.write_file(time_df, save_path, overwrite=overwrite)

    return time_df

def time_stats(time_df,
               save_dir,
               units='minutes',
               condition='all',
               overwrite=False):
    '''
    Calculate mean, std of timings across participants
    '''

    # If file exists, return it
    if save_dir is not None:
        save_path = os.path.join(save_dir, 'timing_stats_{}_{}.csv'.format(condition, units))
        if os.path.exists(save_path) and not overwrite:
            utils.informal_log("Timing statistics exists at {}".format(save_path))
            return utils.read_file(save_path)

    stats_df = get_stats_df(
        df=time_df,
        stats=['mean', 'std', 'median'],
        axis_name='section')

    if save_dir is not None:
        utils.write_file(stats_df, save_path, overwrite=overwrite)

    return stats_df

'''
Functions for Demographics
'''
def get_demographics(df,
                     demographic_qs,
                     save_dir=None,
                     overwrite=False):
    '''
    Given raw dataframe, return dataframe of demographics
    '''
    if save_dir is not None:
        save_path = os.path.join(save_dir, 'demographics.csv')
        if os.path.exists(save_path) and not overwrite:
            utils.informal_log("Demographics exists at {}".format(save_path))
            return utils.read_file(save_path)

    demographics_dfs = []
    for q_id, variable in demographic_qs.items():
        if q_id in df.columns:
            column = df[q_id]
            # Get counts
            value_counts = column.value_counts()

            # Add values as a column
            value_counts = value_counts.reset_index()
            value_counts = value_counts.rename(columns={q_id: 'level'})

            # Add variable name as leftmost column
            value_counts.insert(0, column='variable', value=variable)
            # Compute percent
            total_counts = value_counts['count'].sum()
            value_counts['percent'] = value_counts['count'] / total_counts * 100
        else:
            column_names = df.filter(like=q_id).columns

            utils.informal_log("Column {} not found".format(q_id))
            continue

        demographics_dfs.append(value_counts)

    demographics_df = pd.concat(demographics_dfs)

    if save_dir is not None:
        utils.informal_log("Saving demographics to {}".format(save_path))
        utils.write_file(demographics_df, save_path, overwrite=overwrite)

'''
Save responses to an FRQ
'''
def save_frq(df,
             q_id,
             addit_q_id,
             frq_save_path,
             print_responses=True):
    '''
    Given a dataframe (raw) save the condition, prolific ID and question ID columns
    '''

    frq_df = df.loc[:, [addit_q_id, 'PROLIFIC_PID', q_id]]
    frq_df = frq_df[~frq_df[q_id].isnull()]
    frq_responses = []

    if print_responses:
        utils.informal_log("Printing first 5 responses...")
    for idx, row in frq_df.iterrows():
        frq_responses.append("{} [{}]: \n\t{}".format(
            row['PROLIFIC_PID'], row[addit_q_id], row[q_id]))
        if print_responses and idx < 5:
            utils.informal_log("{} [{}]: \n\t{}".format(
                row['PROLIFIC_PID'], row[addit_q_id], row[q_id]
            ))
    utils.informal_log("\n Saving FRQ responses to {}".format(frq_save_path))
    utils.write_file(frq_responses, frq_save_path)
    return frq_df


'''
Exclusions
'''
def exclude_min_responses(df,
            k,
            n_total=40):
    '''
    Only keep rows with at least k responses

    Arg(s):
        df : pd.DataFrame
        k : int

    Returns pd.DataFrame
    '''
    if k == 0:
        return df

    # Sum up NaNs per column (item)
    n_nan_item = df.isnull().sum(axis=0)
    utils.informal_log("Pre-filtering:")
    for idx, count in n_nan_item.items():
        if count > 0:
            utils.informal_log("{} has {} NaNs".format(idx, count))

    # Sum up number of NaNs per row (participant)
    n_responses = n_total - df.isnull().sum(axis=1)

    df = df[n_responses >= k]

    # Post filtering
    n_nan_item = df.isnull().sum(axis=0)
    utils.informal_log("\nPost-filtering:")
    for idx, count in n_nan_item.items():
        if count > 0:
            utils.informal_log("{} has {} NaNs".format(idx, count))
    return df

def _verify_post_attention_check(df):
    '''
    Given a dataframe, return the rows that pass the attention check
    '''
    # Extract column
    ac_response = df['ACQ1']
    # TODO: Verify correct number of responses (2)
    n_selected = (ac_response.str.count(',') + 1).to_numpy()
    selected_correct = n_selected <= 2

    selection1 = ac_response.str.contains('understanding how others are feeling').to_numpy()
    selection2 = ac_response.str.contains('doing computations').to_numpy()

    pass_ac = selected_correct & selection1 & selection2

    return df[pass_ac]


def perform_exclusions(df,
                       # Exclusion parameters
                       min_survey_time,
                       min_median_per_page_time,
                       post_attention_check=True,
                       manual_exclusions=None):
    '''
    Given raw DF, return rows that meet minimum time requirement and pass the attention check
    '''

    # Drop rows that are not responses
    df = df[df['ResponseId'].str.startswith('R_')]

    # Keep rows that spent at least min_survey_time minutes
    if min_survey_time is not None:
        df = df[df['survey_time'] >= min_survey_time]
        utils.informal_log("{} rows remaining post filtering for minimum of {} minutes on survey".format(
            len(df), min_survey_time))
    else:
        utils.informal_log("No exclusions for minimum time spent on survey performed")

    # Keep rows that spend a median of at least min_median_per_page_time minutes
    if min_median_per_page_time is not None:
        df = df[df['median_time_per_survey_page'] >= min_median_per_page_time]
        utils.informal_log("{} rows remaining post filtering for minimum median of {} minutes on each survey page".format(
            len(df), min_median_per_page_time))
    else:
        utils.informal_log("No exclusions for minimum median amount of time")

    # Keep rows that pass the attention check
    if post_attention_check:
        df = _verify_post_attention_check(df=df)
        utils.informal_log("{} rows remaining post filtering for post-survey attention check correctness".format(
            len(df), min_survey_time))
    else:
        utils.informal_log("No post attention check verification")

    # Remove any manually excluded rows (based on Prolific ID)
    if manual_exclusions is not None:
        df = df[~df['PROLIFIC_PID'].isin(manual_exclusions)]

        utils.informal_log("{} rows remaining post manual filtering for the following Prolific IDs: {}".format(
            len(df), manual_exclusions))
    else:
        utils.informal_log("No manual exclusions performed")

    utils.informal_log("Number of rows: {}".format(len(df)))
    utils.informal_log("Number of rows in baseline: {}".format(len(df[df['CONDITION'] == 'Baseline'])))
    utils.informal_log("Number of rows in mechanistic: {}".format(len(df[df['CONDITION'] == 'Mechanistic'])))
    utils.informal_log("Number of rows in functional: {}".format(len(df[df['CONDITION'] == 'Functional'])))
    utils.informal_log("Number of rows in intentional: {}".format(len(df[df['CONDITION'] == 'Intentional'])))

    return df

'''

ANALYSIS

'''
'''
Functions for Mean Ratings & Statistics of Ratings
'''
def get_mc_mapping(items,
                   n_pages,
                   n_items_per_page,
                   q_id,
                   suffix=None,
                   columns=None):
    '''
    Obtain mapping between data CSV column names and mental capacity attribution names

    Arg(s):
        items : list[str]
            list of items in order that they were put in the loop + merge
    '''

    n_items = len(items)
    assert n_pages * n_items_per_page == n_items, "With {} pages and {} items per page, expected {} items, but received {}".format(
        n_pages, n_items_per_page, n_pages * n_items_per_page, n_items)

    mapping = {}

    for page_idx in range(n_pages):
        for item_idx in range(n_items_per_page):
            if n_items_per_page > 1:
                q_name = '{}_{}_{}'.format(page_idx + 1, q_id, item_idx + 1)
            else:
                q_name = '{}_{}'.format(page_idx + 1, q_id)

            if suffix is not None:
                q_name += "_{}".format(suffix)
            if columns is not None:
                assert q_name in columns, "Question name \'{}\' was not found in columns".format(q_name)

            ms_idx = item_idx * n_pages + page_idx
            ms_name = items[ms_idx]
            mapping[q_name] = ms_name

    return mapping

def _add_groupings(rating_df,
                   groupings):

    # Add columns for each category dimension
    for grouping_source, grouping in groupings.items():
        for category_name, category_mental_states in grouping.items():
            column_name = '{}_{}'.format(grouping_source, category_name)
            if column_name in rating_df.columns:
                continue

            # Get columns that make up this category and calculate mean
            category_df = rating_df[rating_df.columns.intersection(category_mental_states)]
            category_mean = np.nanmean(category_df.to_numpy(), axis=1)

            # Assign to column
            rating_df[column_name] = category_mean
    return rating_df

def get_ratings(df,
                mc_mapping,
                groupings,
                n_total=40,
                save_dir=None,
                overwrite=False):
    '''
    Given raw data, return cleaned ratings for all participants. Additionally add ratings for each category dimension

    Arg(s):
        df : pd.DataFrame
            Raw Qualtrics Data
        mc_mapping : dict[str: str] mental state mapping from Qualtrics name to our name
        groupings : dict[str : dict[str : list[str]]]
            outer keys: weisman/colombatto
            inner keys: category/condition
        save_dir : str or None
        overwrite : bool

    '''
    # If file exists, return it
    if save_dir is not None:
        save_path = os.path.join(save_dir, 'ratings.csv')
        if os.path.exists(save_path) and not overwrite:
            utils.informal_log("Ratings exists at {}".format(save_path))
            return utils.read_file(save_path)

    '''
    Create rating DF for main analysis
    '''
    # Only keep columns that are mental states
    rating_df = df[mc_mapping.keys()]
    rating_df = rating_df.rename(columns=mc_mapping)

    # Since 1, 4, & 7 options have text, convert them to just numbers
    rating_df = rating_df.replace({'1 (Not at all)': '1 (Not at all capable)'})
    rating_df = rating_df.replace({'1 (Not at all capable)': '1', '7 (Highly capable)': '7', '4 (Somewhat capable)':  '4'})
    # Convert data to float
    rating_df = rating_df.map(lambda x: pd.to_numeric(x, errors='raise'))

    # Add mean rating for all items
    items = mc_mapping.values()
    mean_ratings = rating_df[rating_df.columns.intersection(items)].mean(axis=1)
    rating_df['all_items'] = mean_ratings

    # Add category dimensions
    rating_df = _add_groupings(
        rating_df=rating_df,
        groupings=groupings)

    # Save condition
    rating_df['condition'] = df['CONDITION']

    # Save participant ID
    rating_df['participant_id'] = df['PROLIFIC_PID']

    if save_dir is not None:
        utils.write_file(rating_df, save_path, overwrite=overwrite)

    # Print stats of rating_df
    utils.informal_log("Rating DF stats:")
    utils.informal_log("Number of rows: {}".format(len(rating_df)))
    utils.informal_log("Number of rows in baseline: {}".format(len(rating_df[rating_df['condition'] == 'Baseline'])))
    utils.informal_log("Number of rows in mechanistic: {}".format(len(rating_df[rating_df['condition'] == 'Mechanistic'])))
    utils.informal_log("Number of rows in functional: {}".format(len(rating_df[rating_df['condition'] == 'Functional'])))
    utils.informal_log("Number of rows in intentional: {}".format(len(rating_df[rating_df['condition'] == 'Intentional'])))

    return rating_df

def _mean_ratings(rating_df,
                  condition=None,
                  save_dir=None,
                  overwrite=False):
    '''
    Given ratings data, return DF with statistics for each mental state
    '''

    # If file exists, return it
    if save_dir is not None:
        filename = 'rating_stats.csv'
        if condition is not None:
            filename = '{}_{}'.format(condition, filename).lower()
        save_path = os.path.join(save_dir, filename)
        if os.path.exists(save_path) and not overwrite:
            utils.informal_log("Rating statistics exists at {}".format(save_path))
            return utils.read_file(save_path)

    stats_df = {}

    # Drop metadata columns
    if 'condition' in rating_df.columns:
        rating_df = rating_df.drop('condition', axis=1)
    if 'participant_id' in rating_df.columns:
        rating_df = rating_df.drop('participant_id', axis=1)

    stats_df['mental_state'] = rating_df.columns
    # means = rating_df.mean(axis=0) # automatically skips NaN
    means = np.nanmean(rating_df.to_numpy(), axis=0)
    stats_df['mean'] = means

    # stds = rating_df.std(axis=0) # automatically skips NaN
    stds = np.nanstd(rating_df.to_numpy(), axis=0)
    stats_df['std'] = stds

    stats_df = pd.DataFrame(stats_df)

    if save_dir is not None:
        utils.write_file(stats_df, save_path, overwrite=overwrite)

    return stats_df

def mean_ratings(rating_df,
                 save_conditions=True,
                 save_dir=None,
                 overwrite=False):
    if save_conditions:
        assert 'condition' in rating_df.columns

    stats_dfs = {}

    stats_dfs['all'] = _mean_ratings(
        rating_df=rating_df,
        condition=None,
        save_dir=save_dir,
        overwrite=overwrite
    )


    # Save separate CSVs for each condition
    if save_conditions:
        for condition in ['Mechanistic', 'Functional', 'Intentional', 'Baseline']:
            condition_df = rating_df[rating_df['condition'] == condition]
            condition_stats_df = _mean_ratings(
                rating_df=condition_df,
                condition=condition,
                save_dir=save_dir,
                overwrite=overwrite
            )
            stats_dfs[condition.lower()] = condition_stats_df


    return stats_dfs

def create_master_stats(stats_dfs,
                        join_col_name='mental_state',
                        save_dir=None,
                        overwrite=False):
    '''
    Given dictionary of dfs, create a master one
    '''
    # If file exists, return it
    if save_dir is not None:
        save_path = os.path.join(save_dir, 'all_conditions_ratings_stats.csv')
        if os.path.exists(save_path) and not overwrite:
            utils.informal_log("Master ratings exists at {}".format(save_path))
            return utils.read_file(save_path)

    df_list = []
    mental_states = None
    for condition, df in stats_dfs.items():
        mental_states = df[join_col_name]
        df = df.drop(join_col_name, axis=1)
        df = df.add_prefix('{}_'.format(condition), axis=1)
        df.insert(0, join_col_name, mental_states)

        df_list.append(df)

    # Iteratively merge DFs on the join_col_name column
    df = functools.reduce(lambda left, right: pd.merge(left,right,on=join_col_name), df_list)

    if save_dir is not None:
        utils.write_file(df, save_path, overwrite=overwrite)

    return df

'''
Functions for data preparation of mixed effects models
'''
def prepare_R_df(rating_df,
                 groupings,
                 save_dir,
                 separate_groups=True,
                 overwrite=False):
    '''
    1. Convert condition column to numbers
    2. Create separate DFs for each category
    '''
    if separate_groups:  # Save separate CSV for each group (body, mind, heart, experience, intelligence)
        for source, source_group in groupings.items():
            for category_name, category_items in source_group.items():
                # Keep table format participants X items + condition
                key = '{}_{}'.format(source,category_name)
                keep_columns = category_items + ['condition', key]
                df = rating_df[rating_df.columns.intersection(keep_columns)]

                save_path = os.path.join(save_dir, '{}.csv'.format(key))
                utils.write_file(df, save_path, overwrite=overwrite)
    else: # Only save separate CSVs for each grouping method (Weisman, Colombatto)
        path_dictionary = {}
        for source, source_group in groupings.items():
            # Create item -> group mapping
            item_group_mapping = {}
            items_list = []
            for category_name, category_items in source_group.items():
                for item in category_items:
                    item_group_mapping[item.replace(' ', '.')] = category_name
                    items_list.append(item)
            # items_list = list(item_group_mapping.keys())
            items_list_R = list(item_group_mapping.keys())
            # Select items and condition
            keep_columns = items_list + ['condition']

            df = rating_df[rating_df.columns.intersection(keep_columns)]
            # Rename columns to replace ' ' with '.'

            df = df.rename(columns=lambda x: x.replace(' ', '.'))
            # Assign PID
            df.loc[:, 'pid'] = ['pid{}'.format(i + 1) for i in range(len(df))]

            # Convert into long format with columns item, rating, condition, pid
            id_vars = ['condition', 'pid']
            value_vars = items_list_R
            df = pd.melt(
                frame=df,
                id_vars=id_vars,
                value_vars=value_vars,
                var_name='item',
                value_name='rating'
            )

            # Add group as a column
            df.loc[:, 'category'] = df['item'].apply(lambda x : item_group_mapping[x])

            save_path = os.path.join(save_dir, '{}.csv'.format(source))
            utils.write_file(df, save_path, overwrite=overwrite)
            path_dictionary[source] = save_path
        path_dictionary_save_path = os.path.join(save_dir, 'paths.json')
        utils.write_file(path_dictionary, path_dictionary_save_path, overwrite=overwrite)

def copy_groupings(groupings,
                   all_items=None,
                   save_dir=None,
                   overwrite=True):
    '''
    Rewrite groupings so they are stored in separate files for each category & replace ' '' with '.' for R
    '''
    utils.ensure_dir(save_dir)

    if all_items is None:
        accumulate_items = True
    else:
        all_items = [item.replace(' ', '.').replace('-', '.') for item in all_items]
        accumulate_items = False
    for grouping_source, grouping in groupings.items():
        if accumulate_items:
            all_items = []
        for category_name, items in grouping.items():
            key = '{}_{}'.format(grouping_source, category_name)
            items = [item.replace(' ', '.').replace('-', '.') for item in items]

            save_path = os.path.join(save_dir, '{}_items.txt'.format(key))
            utils.write_file(items, save_path, overwrite=overwrite)
            if accumulate_items:
                all_items += items
        # Name the file based on grouping source just to make downstream code easier
        all_items_save_path = os.path.join(save_dir, '{}_items.txt'.format(grouping_source))
        utils.write_file(all_items,all_items_save_path, overwrite=overwrite)

'''
Code for formatting data from EMMeans
'''

def read_emmeans_single_variable(results_path,
                                 grouping_source,
                                 variable_name='portrayal',
                                 variable_values=['Baseline', 'Mechanistic', 'Functional', 'Intentional'],
                                 save_dir=None,
                                 overwrite=True):
    '''
    Given path to R results, copy the EMMeans info for emmeans(list(pairwise ~ condition))
    Only works for 'group' or 'condition'

    Arg(s):
        results_path : str
            path to copied R outputs
        conditions : list[str]
            list of conditions in order
        save_dir : str
        overwrite : bool
    '''
    # Each EMMeans table is 5 rows
    emmeans_length = len(variable_values) + 1

    df = None
    if save_dir is not None:
        csv_save_path = os.path.join(save_dir, '{}_emmeans_{}.csv'.format(
            grouping_source, variable_name))
        if os.path.exists(csv_save_path) and not overwrite:
            df = utils.read_file(csv_save_path)
    if df is None:
        results_list = utils.read_file(results_path)
        line_idx_dict = dict(zip(results_list, range(len(results_list))))

        # Get the start index of the EMMeans information
        emmeans_start = line_idx_dict['$`emmeans of {}`'.format(variable_name)] + 1
        emmeans_list = results_list[emmeans_start:emmeans_start + emmeans_length]

        columns = None
        # group = None
        df_dict = {}
        for line in emmeans_list:
            line = line.split()
            # Skip empty lines
            if len(line) == 0:
                continue
            # Encountered header
            if line[0] == variable_name:
                for col in line:
                    if col not in df_dict:
                        df_dict[col] = []
                columns = line.copy()
            else: # Fill in table
                for idx, column in enumerate(columns):
                    # Try converting to float
                    try:
                        df_dict[column].append(float(line[idx]))
                    except:
                        df_dict[column].append(line[idx])

        df = pd.DataFrame(df_dict)
        if save_dir is not None:
            utils.write_file(df, csv_save_path, overwrite=overwrite)


    # Get data in format needed for analysis.grouped_bar_graphs (dict with 'means', 'errors')
    means = []
    errors = []
    for val in variable_values:
        condition_means = []
        condition_errors = []
        mean = float(df[(df[variable_name] == val)]['emmean'].iloc[0])
        lower_cl = float(df[(df[variable_name] == val)]['lower.CL'].iloc[0])
        upper_cl = float(df[(df[variable_name] == val)]['upper.CL'].iloc[0])

        # Error is Confidence Interval (output is 95% CI)
        error = (upper_cl - lower_cl) / 2

        condition_means.append(mean)
        condition_errors.append(error)

        if variable_name == 'portrayal':
            means.append(condition_means)
            errors.append(condition_errors)
        elif variable_name == 'category':
            means.append(condition_means[0])
            errors.append(condition_errors[0])
        else:
            raise ValueError("Unsupported variable name '{}'".format(variable_name))

    graph_data = {
        'means': means,
        'errors': errors,
    }

    return graph_data, df

def read_emmeans_marginalized_result(results_path,
                                    grouping_source,
                                    conditions=['Baseline', 'Mechanistic', 'Functional', 'Intentional'],
                                    marginalized_var='item',
                                    marginalized_var_values=[],
                                    save_dir=None,
                                    overwrite=True):
    '''
    Given path to copied EMMeans results, store as pandas dictionary

    Arg(s):
        results_path : str
            path to copied R outputs
        grouping_source : str
            weisman or colombatto for save path purposes
        conditions : list[str]
            list of conditions in order
        save_dir : str
        overwrite : bool
    '''

    # Manually set groups
    if marginalized_var_values is None or len(marginalized_var_values) == 0:
        raise ValueError("No value passed for marginalized_var_values")

    # Each group's EMMeans table is 6 rows (including space separator)
    emmeans_length = len(marginalized_var_values) * (len(conditions) + 2)

    df = None
    if save_dir is not None:
        # if grouping_source == "weisman" or grouping_source == "colombatto" or grouping_source == "factor_analysis":
        #     csv_save_path = os.path.join(save_dir, '{}_emmeans_condition_by_group.csv'.format(grouping_source))
        if grouping_source == "body-heart-mind" or grouping_source == "factor_analysis":
            csv_save_path = os.path.join(save_dir, "{}.csv".format(grouping_source))
        elif grouping_source == "item_level": # Assumes item lev
            csv_save_path = os.path.join(save_dir, "item_means.csv")
        elif grouping_source == "factor_analysis":
            csv_save_path = os.path.join(save_dir, 'fa.csv')
        elif grouping_source == "mentioned":
            csv_save_path = os.path.join(save_dir, "mentioned.csv")
        else:
            raise ValueError("grouping_source {} not supported".format(grouping_source))
        if os.path.exists(csv_save_path) and not overwrite:
            df = utils.read_file(csv_save_path)
    results_list = utils.read_file(results_path)
    line_idx_dict = dict(zip(results_list, range(len(results_list))))

    if df is None:
        # Get the start index of the EMMeans information
        emmeans_start = line_idx_dict['$`emmeans of portrayal | {}`'.format(marginalized_var)] + 1
        emmeans_list = results_list[emmeans_start:emmeans_start + emmeans_length]

        columns = None
        group = None
        df_dict = {}
        for line in emmeans_list:
            line = line.split()

            if len(line) == 0:
                continue
            # Encountered data for a new group
            if line[0] == marginalized_var:
                group = line[-1][:-1] # Exclude the semi colon

            # Encountered header for the table
            elif line[0] == 'portrayal':
                for col in line:
                    if col not in df_dict:
                        df_dict[col] = []
                if marginalized_var not in df_dict:
                    df_dict[marginalized_var] = []
                columns = line.copy()
            else: # Fill in table
                for idx, column in enumerate(columns):
                    # Try converting to float
                    try:
                        df_dict[column].append(float(line[idx]))
                    except:
                        df_dict[column].append(line[idx])
                df_dict[marginalized_var].append(group)
        # Add last table
        df = pd.DataFrame(df_dict)

        if save_dir is not None:
            utils.write_file(df, csv_save_path, overwrite=overwrite)

    # Get data in format needed for analysis.grouped_bar_graphs (dict with 'means', 'errors')
    means = []
    errors = []
    for condition in conditions:
        condition_means = []
        condition_errors = []
        for value in marginalized_var_values:
            mean = float(df[(df['portrayal'] == condition) & (df[marginalized_var] == value)]['emmean'].iloc[0])
            lower_cl = float(df[(df['portrayal'] == condition) & (df[marginalized_var] == value)]['lower.CL'].iloc[0])
            upper_cl = float(df[(df['portrayal'] == condition) & (df[marginalized_var] == value)]['upper.CL'].iloc[0])

            # Error is Confidence Interval (output is 95% CI)
            error = (upper_cl - lower_cl) / 2

            condition_means.append(mean)
            condition_errors.append(error)
        means.append(condition_means)
        errors.append(condition_errors)

    graph_data = {
        'means': means,
        'errors': errors,
    }
    # Get group means
    if marginalized_var == 'group':
        group_emmeans_start_idx = line_idx_dict['$`emmeans of group`'] + 2 # Skip the header
        group_emmeans_list = results_list[group_emmeans_start_idx:group_emmeans_start_idx + len(marginalized_var_values)]

        group_means = {}
        for row in group_emmeans_list:
            row = row.split()
            # First column is name of group, second column is mean
            group_name = row[0]
            group_mean = float(row[1])
            group_means[group_name] = group_mean
        graph_data['group_means'] = group_means

    return graph_data, df

def _format_and_pivot_emmeans_df(emmeans_df,
                                 target_column):
    '''
    Arg(s):
        emmeans_df : output of read_emmeans_single_variable()
        target_column : str
            Name of column in which to index

    Returns:
        pivot_df : pd.DataFrame
            pivoted version of emmeans_df, preparation for graphing
    '''
    # Calculate 95% CI Error Values
    emmeans_df['ci_error'] = (emmeans_df['upper.CL'] - emmeans_df['lower.CL']) / 2
    # Keep only graphing relevant columns
    if target_column is not None:
        emmeans_df = emmeans_df[['portrayal', 'emmean', 'ci_error', target_column]]
    else:
        emmeans_df = emmeans_df[['portrayal', 'emmean', 'ci_error']]

    # Cleanup
    emmeans_df = emmeans_df.rename({'emmean': 'mean'}, axis=1)

    # Pivot table
    if target_column is not None:
        emmeans_df[target_column] = emmeans_df[target_column].apply(lambda x : x.replace('.', ' '))
        pivot_df = emmeans_df.pivot_table(
            index=target_column,
            columns='portrayal',
            values=['mean', 'ci_error']
        )
        pivot_df.columns = ['{}-{}'.format(condition, metric) for condition, metric in pivot_df.columns]
        pivot_df.reset_index(inplace=True)
    else:
        pivot_df = emmeans_df
    return pivot_df

'''
Functions for analysis of additional DVs
'''

def get_attitudes(df,
                  mapping,
                  likert_mapping,
                  save_dir=None,
                  overwrite=False):
    '''
    Given raw data, return cleaned ratings for all participants. Additionally add ratings for each category dimension

    Arg(s):
        df : pd.DataFrame
            Raw Qualtrics Data
        mapping : dict[str: str] DV mapping from Qualtrics name to our name

        save_dir : str or None
        overwrite : bool

    '''
    # If file exists, return it
    if save_dir is not None:
        save_path = os.path.join(save_dir, 'attitudes.csv')
        if os.path.exists(save_path) and not overwrite:
            utils.informal_log("Attitudes CSV exists at {}".format(save_path))
            return utils.read_file(save_path)

    # Collect columns that are in mapping.keys()
    attitudes_df = df[mapping.keys()]
    attitudes_df = attitudes_df.rename(columns=mapping)
    # Map Likert scale to numbers
    attitudes_df = attitudes_df.map(lambda x: likert_mapping[x])

    # Add condition and participant ID
    attitudes_df.loc[:, 'condition'] = df.loc[:, 'CONDITION']
    attitudes_df.loc[:, 'participant_id'] = df.loc[:, 'PROLIFIC_PID']

    if save_dir is not None:
        utils.write_file(attitudes_df, save_path, overwrite=overwrite)

    return attitudes_df

def save_r_format(attitudes_df,
                  columns,
                  save_dir,
                  overwrite=False):
    '''
    Take the CSV for attitudes and save it in a format compatible with R
    '''
    for column in columns:
        save_path = os.path.join(save_dir, '{}.csv'.format(column))
        if os.path.exists(save_path) and not overwrite:
            continue

        r_df = attitudes_df.loc[:, (column, 'condition')]
        utils.write_file(r_df, save_path, overwrite=overwrite)

'''

EXPLORATORY ANALYSIS

'''
'''
Code for identifying individuals with high leverage
'''
def _get_leverage(X):
	"""
	Calculates the leverage (diagonal of hat matrix) for a given design matrix X.

	Parameters:
	X: N x D np.array representing the design matrix.

	Returns:
	leverage : N-dim np.array
	"""

	X = np.array(X) # Ensure X is a numpy array
	hat = X @ np.linalg.inv(X.T @ X) @ X.T
	leverage = np.diag(hat)
	return leverage

def _identify_outliers(leverage):
	'''
	Given leverage, return indices of outliers with leverage > 2 * mean
	'''
	mean_leverage = np.mean(leverage)
	outlier_idxs = np.argwhere(leverage > 2 * mean_leverage)
	# Convert into 1D list
	outlier_idxs = list(np.squeeze(outlier_idxs))
	return outlier_idxs

def identify_outliers(rating_df,
                      items,
                      save_dir,
                      overwrite=True):
    '''
    Given rating_df, identify outliers and save list of outlier PIDs
    '''
    if save_dir is not None:
        log_path = os.path.join(save_dir, 'log.txt')
    else:
        log_path = None

    # Compute leverage
    rating_array = rating_df[rating_df.columns.intersection(items)].to_numpy()
    leverage = _get_leverage(rating_array)
    # Extract idxs of rows with high leverage
    outlier_idxs = _identify_outliers(leverage)
    if len(outlier_idxs) == 0:
        utils.informal_log("No outliers with high leverage", log_path)
    else:
        outliers = rating_df.loc[outlier_idxs, :]
        outlier_pids = outliers.loc[:, 'participant_id']
        outlier_leverages = leverage[outlier_idxs]
        utils.informal_log("{} outliers with high leverage. Mean leverage: {:.2f}".format(
        len(outlier_pids), np.mean(leverage)), log_path)
        outlier_df = pd.DataFrame({
            'pid': outlier_pids,
            'leverage': outlier_leverages})
        if save_dir is not None:
            outlier_save_path = os.path.join(save_dir, 'high_leverage.csv')
            if not os.path.exists(outlier_save_path) or overwrite:
                utils.write_file(outlier_df, outlier_save_path)
        return outlier_df

'''
Mechanistic Conceptual Questions
'''
mcq_mapping = {
    'Q38': 'attention', # What is the mechanism that LLMs use to understand the context of a sentence called?
    'Q39': 'sampling', # How do LLMs typically select the next word?
    'Q40': 'data_quantity', # True or False: LLMs need a lot of data in order to learn.
    'Q41': 'generation', # How do LLMs generate text?
}

mcq_correct_answers = {
    'attention': 'Attention',
    'sampling': 'Sampling: Choose one of the words with highest probabilities',
    'data_quantity': 'True',
    'generation': 'By repeatedly predicting the next most likely word'
}

def mechanistic_mcq_analysis(df,
                             save_dir,
                             overwrite=False):
    if save_dir is not None:
        save_path = os.path.join(save_dir, 'mech_accuracy.csv')
        if os.path.exists(save_path) and not overwrite:
            utils.informal_log("MCQ accuracy CSV exists at {}".format(save_path))
            return utils.read_file(save_path)

    mcq_response_df = df.loc[:, list(mcq_mapping.keys()) + ['CONDITION', 'PROLIFIC_PID']]
    mcq_response_df = mcq_response_df.rename(columns=mcq_mapping)
    # Only select rows in mechanistic condition
    mcq_response_df = mcq_response_df[mcq_response_df['CONDITION'] == 'Mechanistic']

    n_rows = len(mcq_response_df)
    mcq_correct_df = {}

    # Make DF for correct/incorrect
    for col, correct_response in mcq_correct_answers.items():
        correct = mcq_response_df[col] == correct_response
        mcq_correct_df[col] = correct

        utils.informal_log("{} question: {:.2f}% of participants answered correctly".format(
            col, correct.sum() / n_rows * 100))
    mcq_correct_df = pd.DataFrame(mcq_correct_df)
    mcq_correct_df = pd.concat([mcq_correct_df, mcq_response_df.loc[:, ['CONDITION', 'PROLIFIC_PID']]], axis=1)

    # Summarize across participants
    participant_accuracies = []
    for _, row in mcq_correct_df.iterrows():
        participant_accuracy = row[list(mcq_correct_answers.keys())].sum() / 4 * 100
        participant_accuracies.append(participant_accuracy)


    mcq_correct_df['accuracy'] = participant_accuracies
    visualizations.histogram(
        np.array(participant_accuracies),
        title='Histogram of Mechanistic MCQ Performance ({} participants)'.format(n_rows),
        xlabel='Accuracy (%)',
        ylabel='Number of Participants')

    if save_dir is not None:
        utils.write_file(mcq_correct_df, save_path, overwrite=overwrite)

    return mcq_correct_df

'''
Compute correlations
'''
def addit_dv_correlations(addit_dv_df,
                          rating_df,
                          addit_dv_list,
                          group_col_names,
                          min_max_scale=False,
                          mech_quiz_df=None,
                          p_threshold=0.05,
                          title=None,
                          save_dir=None,
                          overwrite=False):

    if save_dir is not None:
        if mech_quiz_df is None:
            filename = 'correlations'
        else:
            filename = 'mech_correlations'

        # change name for scaled
        if min_max_scale:
            filename += '_scaled'
        correlation_save_path = os.path.join(save_dir, '{}.json'.format(filename))
        correlation_vis_save_path = os.path.join(save_dir, '{}.pdf'.format(filename))
        if os.path.exists(correlation_save_path) and not overwrite:
            utils.informal_log("Correlations exist at {}".format(correlation_save_path))
            return utils.read_file(correlation_save_path)

    # Get data from ratings (the group means)
    group_means = rating_df[rating_df.columns.intersection(group_col_names + ['participant_id'])]

    # Get ratings from additional DVs
    corr_data = addit_dv_df[addit_dv_df.columns.intersection(addit_dv_list + ['participant_id'])]

    # Merge data for spearman correlation based on participant id
    corr_data = pd.merge(corr_data, group_means, on='participant_id')

    # Get labels for correlation matrix
    labels = addit_dv_list + group_col_names

    if mech_quiz_df is not None:

        # Get mechanistic quiz performance
        mech_quiz_df = mech_quiz_df[['accuracy', 'PROLIFIC_PID']]
        mech_quiz_df = mech_quiz_df.rename({'PROLIFIC_PID': 'participant_id'}, axis=1)
        corr_data = pd.merge(corr_data, mech_quiz_df, how='inner')
        labels.append('mech_acc')

        utils.informal_log("Computing correlations with mechanistic quiz acc. Only using rows in mechanistic condition ({} rows)".format(
            len(corr_data)
        ))


    corr_data = corr_data.drop(columns='participant_id')

    # Scale each column by min/max of the column
    if min_max_scale:
        scaler = MinMaxScaler()
        corr_data = pd.DataFrame(scaler.fit_transform(corr_data), columns=corr_data.columns)

        assert corr_data.min(axis=0).sum() == 0
        assert corr_data.max(axis=0).sum() == len(corr_data.columns)

    corr_matrix, pvals = stats.spearmanr(corr_data)

    # Mask to bold the correlations that are significant (mask out those that are not significant)
    is_significant = pvals < p_threshold
    utils.informal_log("Bolding items whose p-values < {:.7f}".format(p_threshold))
    # Turn correlation data into an easy dictionary
    corr_dict = {}
    print(labels)
    for idx1 in range(len(labels) - 1):
        for idx2 in range(idx1 + 1, len(labels)):
            dv1 = labels[idx1]
            dv2 = labels[idx2]

            if dv1 < dv2:
                key = '{}-{}'.format(dv1, dv2)
            else:
                key = '{}-{}'.format(dv2, dv1)

            corr_dict[key] = {
                'spearman': corr_matrix[idx1][idx2],
                'pval': pvals[idx1][idx2]
            }

    # Plot the correlation grid
    plt.figure(figsize=(8, 6))
    # Non significant correlations
    sns.heatmap(
        corr_matrix,
        mask=~is_significant,
        annot_kws={"weight": "bold"},
        vmin=0.0,
        vmax=1.0,
        annot=True,
        cmap='coolwarm',
        fmt=".2f",
        linewidths=.5)
    # Significant correlations
    sns.heatmap(
        corr_matrix,
        mask=is_significant,
        vmin=0.0,
        vmax=1.0,
        annot=True,
        cmap='coolwarm',
        fmt=".2f",
        linewidths=.5,
        xticklabels=labels,
        yticklabels=labels,
        # Avoid having two colorbars
        cbar=False)
    # Add title
    if title is not None:
        plt.title(title)
    # Save
    if save_dir is not None:
        plt.savefig(correlation_vis_save_path, bbox_inches="tight")
        utils.write_file(corr_dict, correlation_save_path, overwrite=overwrite)

    plt.show()
    plt.clf()

    return corr_dict

'''
Inter-Rater Reliability and Inter-Item Reliability
'''
def _calculate_irr(rating_df,
                   dv_items):
    '''
    Return the Fleiss Kappa, Krippendorff's Alpha

    Arg(s):
        rating_df : pd.DataFrame
            rows are participants
            columns are DV items (+ other metadata)
    '''

    # Drop columns that are not rating items
    rating_df = rating_df[rating_df.columns.intersection(dv_items)]

    # Transpose rating_df such that the rows are items and the columns are each rater
    rating_array = rating_df.to_numpy()
    rating_array = rating_array.T

    # Calculate Fleiss' Kappan d Krippendorff's Alpha across all ratings
    fleiss_array, fleiss_categories = irr.aggregate_raters(rating_array)
    fleiss_kappa = irr.fleiss_kappa(table=fleiss_array)
    kripps_alpha = krippendorffs_alpha(fleiss_array)

    results = {
        'fleiss_kappa': fleiss_kappa,
        'krippendorffs_alpha': kripps_alpha}

    return results

def calculate_irr(rating_df,
                  dv_items,
                  conditions,
                  groupings,
                  save_dir,
                  overwrite=True):

    if save_dir is not None:
        save_path = os.path.join(save_dir, 'interrater.json')
        if os.path.exists(save_path) and not overwrite:
            return utils.read_file(save_path)

    assert 'condition' in rating_df.columns


    results = {}
    for grouping_source, grouping in groupings.items():
        for category_name, items in grouping.items():
            key = '{}_{}'.format(grouping_source, category_name)
            category_results = {}
            category_df = rating_df[rating_df.columns.intersection(items + ['condition'])]
            # print(key, category_df.columns)
            # Calculate IRR within each condition
            for condition in conditions:
                assert category_df['condition'].str.contains(condition).any()

                condition_rating_df = category_df[category_df['condition'] == condition]
                condition_results = _calculate_irr(
                    rating_df=condition_rating_df,
                    dv_items=dv_items)
                for name, value in condition_results.items():
                    category_results['{}_{}'.format(condition, name)] = value

            # Calculate IRR overall
            overall_results = _calculate_irr(
                rating_df=category_df,
                dv_items=dv_items)
            for name, value in overall_results.items():
                category_results['overall_{}'.format(name)] = value
            results[key] = category_results

    # Calculate IRR across all categories
    all_category_results = {}
    for condition in conditions:
        condition_rating_df = rating_df[rating_df['condition'] == condition]
        condition_results = _calculate_irr(
            rating_df=condition_rating_df,
            dv_items=dv_items)
        for name, value in condition_results.items():
            all_category_results['{}_{}'.format(condition, name)] = value

    # Calculate IRR overall
    overall_results = _calculate_irr(
        rating_df=rating_df,
        dv_items=dv_items)
    for name, value in overall_results.items():
        all_category_results['overall_{}'.format(name)] = value

    results['all_category'] = all_category_results
    if save_dir is not None:
        utils.write_file(results, save_path, overwrite=overwrite)

    return results

def _calculate_iir(rating_df,
                   dv_items):
    '''
    Calculate inter ITEM reliability using Cronbach's alpha based on the DV items provided

    Arg(s):
        rating_df : pd.DataFrame
            rows are participants
            columns are DV items (+metadata)
        dv_items : list[str]
    '''
    rating_df = rating_df[rating_df.columns.intersection(dv_items)]
    cronbachs_alpha = pg.cronbach_alpha(data=rating_df)
    return cronbachs_alpha

def calculate_iir(rating_df,
                  dv_itemss,
                  conditions,
                  save_dir,
                  overwrite=True):
    '''
    Calculate inter ITEM reliability using Cronbach's alpha
    for each list of dv items in dv_itemss and for all conditions (as well as overall)

    Arg(s):
        rating_df : pd.DataFrame
            rows are participants
            columns are DV items (+metadata)
        dv_itemss : list[list[str] or str]
            if item is str, use all items that have str in it
    '''
    if save_dir is not None:
        save_path = os.path.join(save_dir, 'interitem.json')
        if os.path.exists(save_path) and not overwrite:
            return utils.read_file(save_path)

    results = {}
    for obj in dv_itemss:
        obj_results = {}
        if type(obj) == list:
            dv_items = obj
        elif type(obj) == str:
            dv_items = [col for col in rating_df.columns if obj in col]
        key = '-'.join([item.replace(' ', '_') for item in dv_items])

        # Calculate IIR for this set for all conditions
        for condition in conditions:
            condition_df = rating_df[rating_df['condition'] == condition]
            alpha, ci = _calculate_iir(
                rating_df=condition_df,
                dv_items=dv_items)
            obj_results[condition] = {
                'cronbachs_alpha': float(alpha),
                'ci': ci.tolist()
            }

        # Calculate IIR overall for this set
        alpha, ci = _calculate_iir(
            rating_df=rating_df,
            dv_items=dv_items)

        obj_results['overall'] = {
            'cronbachs_alpha': float(alpha),
            'ci': ci.tolist()
        }
        results[key] = obj_results

    if save_dir is not None:
        utils.write_file(results, save_path, overwrite=overwrite)

    return results

'''

VISUALIZATIONS

'''
'''
Visualize factor loadings
'''
# Assign factor categories
def assign_categories(df,
                      item_colname,
                      save_dir=None,
                      overwrite=True):
    if save_dir is not None:
        save_path = os.path.join(save_dir, 'loading_w_factors.csv')
        dict_save_path = os.path.join(save_dir, 'fa_groupings.json')
        if os.path.exists(save_path) and \
            dict_save_path and \
            not overwrite:
            utils.informal_log("Files exists in {}".format(save_dir))
            return utils.read_file(save_path), utils.read_file(dict_save_path)

    items = df[item_colname]
    factor_df = df.drop(columns=item_colname)
    factor_df['factor'] = factor_df.idxmax(axis=1)
    factor_df[item_colname] = items

    groupings_dict = {}
    for factor in sorted(factor_df['factor'].unique()):
        row_items = factor_df[factor_df['factor'] == factor]['item'].to_list()
        groupings_dict[factor] = row_items

    if save_dir is not None:
        utils.write_file(factor_df, save_path, overwrite=overwrite)
        utils.write_file({'factor_analysis': groupings_dict}, dict_save_path, overwrite=overwrite)
    return factor_df, groupings_dict

def sort_by_loadings(loading_df,
                     factors,
                     factor_colname='factor',
                     item_colname='item'):
    '''
    Factors should actually be in REVERSE order because of how the heatmap presents items
    '''
    item_order = []
    for factor in factors:
        temp = loading_df[loading_df[factor_colname] == '{}'.format(factor)]
        temp = temp.sort_values(by='{}'.format(factor), axis=0, ascending=True)
        item_order += temp[item_colname].to_list()

    return item_order

def visualize_loadings(loading_df,
                       n_components,
                       orientation='vertical',
                       item_colname='item',
                       keepcol_name='Factor',
                       keep_columns=None,
                       item_order=None,
                       save_dir=None,
                       filename=None,
                       save_ext='pdf',
                       overwrite=True):
    if keep_columns is None:
        keep_columns = ['{}{}'.format(keepcol_name, i) for i in range(1, n_components + 1)]

    # Reorder columns
    loading_df = loading_df[keep_columns + ['item']]

    if item_order is None:
        # Sort by increasing loadings starting with F1 -> Fn
        loading_df = loading_df.sort_values(
            by=keep_columns,
            axis=0,
            ascending=False)

        yticklabels = loading_df[item_colname].to_list()
    else:
        # Get order for items and sort DF by order
        order_mapping = {item: rank for rank, item in enumerate(item_order)}
        loading_df['rank'] = loading_df[item_colname].map(order_mapping)
        loading_df = loading_df.sort_values('rank', ascending=False) #.drop(columns=['rank'])
        loading_df = loading_df.drop(columns='rank')

        print(loading_df.columns)
        # Get ytick labels based on order
        yticklabels = loading_df[item_colname].to_list()
    loading_data = loading_df[loading_df.columns.intersection(keep_columns)]

    if orientation == 'vertical':
        plt.figure(figsize=(3, 12))
        sns.heatmap(
            loading_data,
            yticklabels=yticklabels,
            cmap='BuGn',
            annot=True,
            fmt=".2f")
    else:
        loading_data = loading_data.T
        plt.figure(figsize=(25,3))
        xticklabels = yticklabels
        sns.heatmap(
            loading_data,
            xticklabels=xticklabels,
            cmap='BuGn',
            annot=True,
            fmt=".2f")
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)

    if save_dir is not None:
        if filename is None:
            filename = 'loadings_{}'.format(orientation)
        save_path = os.path.join(save_dir, '{}.{}'.format(filename, save_ext))
        if os.path.exists(save_path) and not overwrite:
            utils.informal_log("File at {} exists and not overwriting")
        else:
            plt.savefig(save_path, bbox_inches="tight")
            utils.informal_log("Saved file to {}".format(save_path))

    plt.show()
    plt.clf()

'''
Graphing Helper functions for figures.ipynb
'''
# Helper Functions
def overall_pointplot(r_results_path,
                      grouping_source,
                      conditions,
                      emmeans_graph_save_dir,
                      condition_color_idxs,
                      orientation,
                      marker_size=6,
                      spacing_multiplier=0.1,
                      label=True,
                      show_xlabel=True,
                      show_ylabel=True,
                      show_legend=False,
                      title=None,
                      fig=None,
                      ax=None,
                      font_size_dict={},
                      save_path=None,
                      save_ext='pdf',
                      show=False):
    '''
    Plot mean over all 40 mental capacity items
    '''
    # Parse EMMeans output from R & pivot data
    emmeans_graph_data, emmeans_df = read_emmeans_single_variable(
        results_path=r_results_path,
        grouping_source=grouping_source,
        variable_name='portrayal',
        variable_values=conditions,
        save_dir=emmeans_graph_save_dir,
        overwrite=False)

    emmeans_df = _format_and_pivot_emmeans_df(
        emmeans_df=emmeans_df,
        target_column=None
    )

    # Extract means and CI errors
    means = emmeans_df['mean']
    means = [[mean] for mean in means]

    errors = emmeans_df['ci_error'].to_numpy()
    errors = [[error] for error in errors]

    # Formatting for horizontal vs vertical graphs (based on direction of error bars)
    if orientation == 'horizontal':
        ytick_labels = ['']
        if show_ylabel:
            ylabel = 'Overall Item Mean'
        else:
            ylabel = None
        xtick_labels = [i for i in range(1, 8)]
        if show_xlabel:
            xlabel = 'Rating (1-7)'
        else:
            xlabel=None
        fig_size = (7, 3)
    else:
        xtick_labels = ['Overall']
        if show_xlabel:
            if fig is None and ax is None:
                xlabel = 'Overall Item Mean'
            else:
                xlabel = 'Overall'
        else:
            xlabel = None
        ytick_labels = [i for i in range(1, 8)]
        if show_ylabel:
            ylabel = 'Rating (1-7)'
        else:
            ylabel = None
        fig_size = (2, 4)
        xlim = (-1, 1)

    # Label conditions
    if label:
        labels = conditions
    else:
        labels = None

    fig, ax = visualizations.pointplot(
        fig=fig,
        ax=ax,
        means=means,
        errors=errors,
        orientation=orientation,
        labels=labels,
        show_legend=show_legend,
        ytick_labels=ytick_labels,
        yticks=ytick_labels,
        ylim=(0.5, 7.5),
        xtick_labels=xtick_labels,
        xlabel=xlabel,
        ylabel=ylabel,
        xlim=xlim,
        title=title,
        legend_loc='upper left',
        fig_size=fig_size,
        color_idxs=condition_color_idxs,
        marker_size=marker_size,
        show_grid=True,
        spacing_multiplier=spacing_multiplier,
        font_size_dict=font_size_dict,
        save_path=save_path,
        show=show)
    return fig, ax

def category_level_pointplot(r_results_path,
                      grouping_source,
                      groups,
                      conditions,
                      emmeans_graph_save_dir,
                      condition_color_idxs,
                      orientation,
                      show_xlabels,
                      show_ylabels,
                      marker_size,
                      label=False,
                      show_legend=True,
                      title=None,
                      fig=None,
                      ax=None,
                      font_size_dict={},
                      show=False,
                      save_path=None):
    '''
    Plots means of each category
    '''

    # Parse EMMeans output from R & pivot data
    emmeans_graph_data, emmeans_df = read_emmeans_marginalized_result(
        results_path=r_results_path,
        grouping_source=grouping_source,
        conditions=conditions,
        marginalized_var='category',
        marginalized_var_values=groups,
        save_dir=emmeans_graph_save_dir,
        overwrite=True
    )
    pivot_df = _format_and_pivot_emmeans_df(
        emmeans_df=emmeans_df,
        target_column='category'
    )

    # Extract means and 95% confidence intervals
    means = pivot_df[["mean-{}".format(condition) for condition in conditions]].to_numpy().T
    errors = pivot_df[["ci_error-{}".format(condition) for condition in conditions]].to_numpy().T

    # Formatting for horizontal vs vertical graphs (based on direction of error bars)
    if orientation == 'horizontal':
        if show_ylabels:
            ytick_labels = [label.capitalize() for label in pivot_df['category'].to_list()]
            ylabel = 'Item Categories'
        else:
            ytick_labels = None
            ylabel = None

        if show_xlabels:
            xtick_labels = [i for i in range(1, 8)]
            xlabel = 'Rating (1-7)'
        else:
            xtick_labels = None
            xlabel = None
        fig_size = (7, 3)
    else:
        xtick_labels = [label.capitalize() for label in pivot_df['category'].to_list()]
        if show_xlabels:
            if fig is None and ax is None:
                xlabel = 'Item Categories'
            else:
                xlabel = 'Item Categories\n(b)'
        else:
            xlabel = None

        if show_ylabels:
            ytick_labels = [i for i in range(1, 8)]
            ylabel = 'Rating (1-7)'
        else:
            ytick_labels = None
            ylabel = None

        fig_size = (3, 4)
        xlim = (-0.5, 2.5)

    # Label with conditions or not
    if label:
        labels = conditions
    else:
        labels = None

    fig, ax = visualizations.pointplot(
        fig=fig,
        ax=ax,
        means=means,
        errors=errors,
        orientation=orientation,
        labels=labels,
        show_legend=show_legend,
        ytick_labels=ytick_labels,
        yticks=ytick_labels,
        ylim=(0.5, 7.5),
        ylabel=ylabel,
        xtick_labels=xtick_labels,
        xlabel=xlabel,
        xlim=xlim,
        title=title,
        legend_loc='upper left',
        fig_size=fig_size,
        color_idxs=condition_color_idxs,
        marker_size=marker_size,
        font_size_dict=font_size_dict,
        show_grid=True,
        save_path=save_path,
        show=show)

    return fig, ax