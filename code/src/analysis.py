from agreement.metrics import krippendorffs_alpha
import functools
# import math
import matplotlib.pyplot as plt
import matplotlib.transforms as transforms
import numpy as np
import os, #sys
import pandas as pd
import pingouin as pg
import scipy.stats as stats
import seaborn as sns
# from sklearn.decomposition import PCA, FactorAnalysis, NMF
# from sklearn.linear_model import LinearRegression
# from sklearn.metrics import mean_squared_error
from statsmodels.stats import inter_rater as irr
from sklearn.preprocessing import MinMaxScaler
# import statsmodels.api

# sys.path.insert(0, 'src')
import utils
import visualizations

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
    for _, row in frq_df.iterrows():
        utils.informal_log("{} [{}]: \n\t{}".format(
            row['PROLIFIC_PID'], row[addit_q_id], row[q_id]
        ), frq_save_path)

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

# def _verify_insurvey_attention_check(df):
#     '''
#     Given dataframe, return rows that pass in survey attention check
#     '''
#     ac_response = df['41_Q35']
#     pass_ac = ac_response.str.contains('4')

#     return df[pass_ac]

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
Functions for plotting bar graphs
'''
# def jitter_dots(dots):
#     '''
#     Arg(s):
#         dots : return object of plt.scatter()
#     '''
#     offsets = dots.get_offsets()
#     jittered_offsets = offsets
#     # only jitter in the x-direction
#     jittered_offsets[:, 0] += np.random.uniform(-0.05, 0.05, offsets.shape[0])
#     dots.set_offsets(jittered_offsets)

#     return dots

# def jitter1D(x,
#            y,
#            jitter_x=True,
#            jitter_y=False):
#     '''
#     Given 1D arrays x, y apply jitter
#     '''
#     assert len(x) == len(y)
#     if jitter_x:
#         x += np.random.uniform(-0.05, 0.05, x.shape)
#     if jitter_y:
#         y += np.random.uniform(-0.05, 0.05, y.shape)
#     return x, y

# def _get_errors(df,
#                 error_dim,
#                 dim_axis_dict):
#     assert error_dim in dim_axis_dict
#     axis = dim_axis_dict[error_dim]
#     if axis is None:
#         error_df = df.to_numpy()
#     else:
#         error_df = np.nanmean(df.to_numpy(), axis=axis)
#     std = np.nanstd(error_df)
#     sem = stats.sem(error_df, axis=None, nan_policy='omit')
#     ci = stats.t.interval(
#         confidence=0.95,
#         df=len(error_df)-1,
#         loc=np.mean(error_df),
#         scale=sem)

#     assert np.abs(np.mean(ci) - np.mean(error_df)) < 1e-5

#     return float(std), float(sem), ci

# def prep_graph_data(rating_df,
#                     groupings,
#                     grouping_source,
#                     all_items,
#                     ci_dim,
#                     jitter_dim,
#                     conditions,
#                     # Axes labels
#                     title=None,
#                     save_dir=None,
#                     overwrite=True,
#                     debug=False):
#     '''
#     Prepare data from rating DF to be used for grouped_bar_graphs
#     '''
#     assert ci_dim in ['participants', 'items', 'both']
#     assert jitter_dim in ['participants', 'items', 'both', None]

#     groups = list(groupings.keys())
#     participant_conditions = rating_df['condition']
#     # condition_set = ['Baseline', 'Mechanistic', 'Functional', 'Intentional'] # list(set(participant_conditions))

#     graph_data = []  # n_conditions x n_groups array
#     dim_axis_dict = {
#         'participants': 1,
#         'items': 0,
#         'both': None
#     }
#     errors = []  # n_conditions x n_groups array for CI error bars
#     jitter_ys = []  # (n_conditions * n_groups) x variable_length array
#     group_means = []  # n_groups array

#     results = {}
#     if all_items is not None:
#         condition_only_results = {
#             'means': [],
#             'errors': []
#         }
#         # Select only columns that are rating items
#         rating_df = rating_df[rating_df.columns.intersection(all_items)]

#     for condition in conditions:
#         # Select  only rows that belong to this condition
#         condition_df = rating_df[participant_conditions == condition] # Select rows
#         condition_results = {}
#         if all_items is not None:

#             # Get mean, errors across all categories for this condition
#             c_mean = np.nanmean(condition_df.to_numpy())
#             std, sem, ci = _get_errors(
#                 df=condition_df,
#                 error_dim=ci_dim,
#                 dim_axis_dict=dim_axis_dict)
#             ci_error = (ci[1] - ci[0]) / 2

#             condition_only_results[condition] = {
#                 'mean': c_mean,
#                 'std_{}'.format(ci_dim): std,
#                 'sem_{}'.format(ci_dim): sem,
#                 'ci_{}'.format(ci_dim): ci,
#                 'ci_error_{}'.format(ci_dim): ci_error
#             }
#             condition_only_results['means'].append([c_mean])
#             condition_only_results['errors'].append([ci_error])

#         # Get means, errors for each category in this condition
#         condition_means = []
#         condition_errors = []
#         condition_jitters = []


#         for group_name, group_mental_states in groupings.items():
#             condition_group_results = {}
#             condition_group_df = condition_df[rating_df.columns.intersection(group_mental_states)]
#             # Check that the number of columns in resulting df is same as number of items in group
#             assert len(group_mental_states) == len(condition_group_df.columns), "Expected {} items in group, only filtered out {} in DF".format(
#                 len(group_mental_states), len(condition_group_df.columns))

#             # condition_means.append(condition_group_df.mean(axis=None, skipna=True))
#             mean = np.nanmean(condition_group_df.to_numpy())
#             condition_means.append(mean)
#             condition_group_results['mean'] = mean

#             # Calculate 95% CI
#             std, sem, ci = _get_errors(
#                 df=condition_group_df,
#                 error_dim=ci_dim,
#                 dim_axis_dict=dim_axis_dict)
#             condition_group_results['std_{}'.format(ci_dim)] = std
#             condition_group_results['sem_{}'.format(ci_dim)] = sem
#             condition_group_results['ci_{}'.format(ci_dim)] = ci

#             # Calculate CI error amount
#             ci_error = (ci[1] - ci[0]) / 2
#             condition_errors.append(ci_error)

#             if jitter_dim is None:
#                 pass
#             elif jitter_dim == 'participants':
#                 condition_jitters.append(np.nanmean(condition_group_df.to_numpy(), axis=1).tolist())

#             elif jitter_dim == 'items':
#                 condition_jitters.append(np.nanmean(condition_group_df.to_numpy(), axis=0).tolist())

#             elif jitter_dim == 'both':
#                 condition_jitters.append(condition_group_df.to_numpy().flatten().tolist())

#             condition_results[group_name] = condition_group_results

#         # Append to outer lists
#         graph_data.append(condition_means)
#         errors.append(condition_errors)
#         jitter_ys.append(condition_jitters)

#         results[condition] = condition_results

#     # Calculate means for each group
#     results['group_means'] = {}
#     group_items = []
#     for group_name, group_mental_states in groupings.items():
#         # Select columns that belong to this group
#         group_df = rating_df[rating_df.columns.intersection(group_mental_states)]
#         # Check that the number of columns in resulting df is same as number of items in group
#         assert len(group_mental_states) == len(group_df.columns), "Expected {} items in group, only filtered out {} in DF".format(
#             len(group_mental_states), len(group_df.columns))
#         # Calculate the mean across conditions for this group
#         group_mean = np.nanmean(group_df.to_numpy())
#         group_means.append(group_mean)

#         results['group_means'][group_name] = group_mean
#         group_items.append(group_df.columns.to_list())

#     # Add to results object
#     results['means'] = graph_data
#     results['errors'] = errors
#     results['jitter_ys'] = jitter_ys
#     results['group_items'] = group_items

#     # Save JSON results
#     if save_dir is not None:
#         if debug:
#             save_dir = os.path.join('debug', save_dir)

#         results_save_path = os.path.join(save_dir, 'group_graph_{}-ci_{}_data.json'.format(
#             grouping_source, ci_dim
#         ))

#         utils.write_file(results, results_save_path, overwrite=overwrite)
#         if all_items is not None:
#             condition_only_results_save_path = os.path.join(
#                 save_dir,
#                 'graph_condition_data.json'
#             )
#             utils.write_file(condition_only_results, condition_only_results_save_path, overwrite=overwrite)

#     if all_items is not None:
#         return results, condition_only_results
#     else:
#         return results

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

# def grouped_bar_graphs(groups,
#                        grouping_source,
#                        conditions,
#                        graph_data,
#                        ci_dim,
#                        jitter_dim,
#                        bar_alpha=0.75,
#                        # Parameters for graphing lines
#                        annotate_items=None,
#                        line_start=0,
#                        line_n_show=5,
#                        color_idxs=[7, 1, 2, 4],
#                        fig_size=None,
#                        title=None,
#                        save_dir=None,
#                        filename=None,
#                        save_ext='png',
#                        overwrite=True,
#                        debug=False):
#     '''
#     Given groupings of what mental state is in which group and rating DF, make Fig 1 style bar graph
#     Group is like Body, Heart, Mind from Weisman or Experience, Intelligence from Colombatto

#     Arg(s):
#         groupings : dict[str : list[str]]
#             dictionary of dimension name -> list of mental states
#         grouping_source : str
#             'weisman' or 'colombatto'
#         rating_df : n_participants X n_mental_states + 1 pd.DataFrame
#         ci_dim : str
#             variable to compute CI over
#             'participants', 'items', or 'both'
#         jitter_dim : str or None
#             variable to plot jitter over
#             'participants', 'items', or 'both'
#         save_dir : str
#         save_ext : str
#         debug : bool
#     '''

#     # Extract data
#     means = np.array(graph_data['means'])
#     errors = np.array(graph_data['errors'])

#     # Display parameters for not marginalized by group
#     if groups is None or len(groups) == 0:
#         if title is None:
#             title = "Mean Ratings Across Conditions ({})".format(grouping_source.capitalize())
#         groups = []
#         ylim = [0, 6]
#         if fig_size is None:
#             fig_size = (6, 6)
#         legend_loc = 'upper right'
#     else:
#         if title is None:
#             title = "Mean Ratings Across Conditions for Each {} Category".format(grouping_source.capitalize())
#         ylim = [0, 7]
#         if fig_size is None:
#             fig_size = (7, 5)
#         legend_loc = 'upper right'
#     # Axis Labels

#     xlabel = ""
#     ylabel = "Rating (1-7)"


#     yticks = [i for i in range(1, min(8, math.floor(ylim[1]) + 1))]
#     yticklabels = [str(i) for i in yticks]

#     # Plot bar graph on figure

#     fig, ax, adjusted_xpos = visualizations.bar_graph(
#         data=means,
#         errors=errors,
#         groups=conditions,
#         labels=[group.capitalize() for group in groups],
#         separate_legends=True,
#         legend_loc=legend_loc,
#         alpha=bar_alpha,
#         title=title,
#         xlabel=xlabel,
#         ylabel=ylabel,
#         ylim=ylim,
#         yticks=yticks,
#         yticklabels=yticklabels,
#         return_adjusted_xpos=True,
#         fig_size=fig_size,
#         color_idxs=color_idxs,
#         show=False
#     )

#     # Extract group means
#     if 'group_means' in graph_data:
#         group_means = graph_data['group_means']

#         # Draw horizontal line for group mean
#         increment = 1 / len(group_means)
#         for idx, group_mean in enumerate(group_means.values()):
#             start = idx * increment
#             end = start + increment
#             ax.axhline(
#                 group_mean,
#                 xmin=start,
#                 xmax=end)

#     # Perform jittering & add to plot
#     if jitter_dim is not None:
#         assert 'jitter_ys' in graph_data
#         assert 'group_items' in graph_data
#         jitter_ys = graph_data['jitter_ys']
#         group_items = graph_data['group_items']

#         n_groups = len(jitter_ys[0])
#         if n_groups == 2:
#             legend_locs = ['upper left', 'upper right']
#         elif n_groups == 3:
#             legend_locs = ['upper left', 'upper center', 'upper right']
#         else:
#             raise ValueError("{} groups not yet supported".format(n_groups))

#         for group_idx in range(n_groups):
#             group_points = []
#             for condition_idx in range(len(jitter_ys)):
#                 x_center = adjusted_xpos[condition_idx][group_idx]
#                 jitter_y = jitter_ys[condition_idx][group_idx]
#                 # Repeat x values to complement y's
#                 x = np.full(len(jitter_y), x_center)

#                 # Jitter x-values
#                 x, y = jitter1D(
#                     x=x,
#                     y=jitter_y,
#                     jitter_x=True,
#                     jitter_y=False)

#                 group_points.append(np.stack([x, y], axis=1))

#             # Conver to numpy array and transpose
#             group_points = np.array(group_points)  # N_conditions X N_items_in_group X 2
#             group_points = np.swapaxes(group_points, 0, 1) # N_items_in_group X N_conditions X 2


#             # Separate X and Y
#             line_xs = group_points[..., 0]
#             line_ys = group_points[..., 1]

#             # Annotations for each line
#             cur_group_items = group_items[group_idx]
#             annotations = []
#             for item in cur_group_items:
#                 annotations.append([item, "","",""])

#             if line_start is None or line_start >= len(cur_group_items):
#                 continue

#             # If pass in items to annotate, that takes precedence over indices
#             if annotate_items is not None:
#                 select_idxs = np.array([True if item in annotate_items else False for item in cur_group_items])
#                 xs = line_xs[select_idxs]
#                 ys = line_ys[select_idxs]
#                 labels = list(np.array(cur_group_items)[select_idxs])
#                 labels = [str(label) for label in labels]
#             else:
#                 xs = line_xs[line_start:line_start+line_n_show]
#                 ys = line_ys[line_start:line_start+line_n_show]
#                 labels = cur_group_items[line_start:line_start+line_n_show]
#             # Plot lines
#             fig, ax = visualizations.plot(
#                 xs=xs,
#                 ys=ys,
#                 fig=fig,
#                 ax=ax,
#                 separate_legends=True,
#                 legend_loc=legend_locs[group_idx],
#                 marker_size=3,
#                 alpha=1,
#                 labels=labels,
#                 scatter=True,
#                 line=True)

#             lines = ax.get_lines()

#     if save_dir is not None:
#         if debug:
#             save_dir = os.path.join('debug', save_dir)
#         utils.ensure_dir(save_dir)
#         # Save graph
#         if filename is None:
#             filename = ""
#             if len(groups) > 0:
#                 filename += "group_"
#             filename += "graph_{}".format(grouping_source)
#             if jitter_dim is not None:
#                 filename += "-jitter_{}".format(jitter_dim)
#             if ci_dim is not None:
#                 filename += "-ci_{}".format(ci_dim)
#             if annotate_items is None:
#                 if line_start is not None:
#                     filename += "-items_{}_{}".format(line_start, line_start + line_n_show)
#             else:
#                 filename += "-items_custom_{}".format(len(annotate_items))

#         filename += ".{}".format(save_ext)
#         save_path = os.path.join(save_dir, filename)
#         if not os.path.exists(save_path) or overwrite:
#             plt.savefig(save_path, bbox_inches='tight')
#             utils.informal_log("Saved graph to {}".format(save_path))
#         else:
#             utils.informal_log("Path at {} exists and not overwriting".format(save_path))
#     plt.show()

'''
N x N Correlation Matrix
'''
def correlation_matrix(rating_df,
                       group_items,
                       save_dir=None):
    condition_order = {
        'Baseline': 0,
        'Mechanistic': 1,
        'Functional': 2,
        'Intentional': 3
    }
    line_color = 'black'
    n_per_condition = len(rating_df) / len(condition_order)

    # Get number per condition
    n_per_condition_dict = {}
    for condition in condition_order.keys():
        print(condition)
        n_per_condition_dict[condition] = len(rating_df[rating_df['condition'] == condition])

    # Sort by condition, grouping conditions together
    rating_df = rating_df.sort_values(by='condition', key=lambda x: x.map(condition_order))
    rating_df = rating_df.reset_index(drop=True)
    condition_pid_df = rating_df.loc[:, ['condition', 'participant_id']]
    assert len(set(condition_pid_df['participant_id'])) == len(condition_pid_df['participant_id'])

    rating_df = rating_df.drop(labels=['condition','weisman_body', 'weisman_heart', 'weisman_mind',
       'colombatto_experience', 'colombatto_intelligence', 'participant_id'], axis=1)

    # If pass in a list of items, select these columns
    if group_items is not None:
        rating_df = rating_df[group_items]

    # Convert into an N x K numpy array
    # ratings = rating_df.to_numpy()

    corr = rating_df.transpose().corr('spearman')
    print(corr.shape)

    # Plot correlation matrix w/colorbar
    im = plt.matshow(corr)
    plt.colorbar(im)

    # Add title
    plt.title("Spearman Correlation Between Participants", y=1.1)

    # Annotations for Condition
    n_running_responses = 0
    condition_pid_dict = {}
    for condition_name in condition_order.keys():
        # Calculate positions for text and dividers
        n_per_condition = n_per_condition_dict[condition_name]
        divider_pos = n_running_responses + n_per_condition - 0.5
        text_pos = (n_running_responses * 2 + n_per_condition) / 2 - 0.5

        # X-Axis annotations
        plt.annotate(condition_name, xy=(text_pos, 0), xytext=(text_pos, -2),
                     ha='center')
        plt.annotate('', xy=(divider_pos, 0), xytext=(divider_pos, -5),
                 arrowprops=dict(arrowstyle="-", color=line_color))
        plt.axvline(x=divider_pos, color=line_color, linestyle='-')

        # Y-Axis annotations
        plt.annotate(condition_name, xy=(0, text_pos), xytext=(-3, text_pos),
                     va='center', rotation=90)
        plt.annotate('', xy=(0, divider_pos), xytext=(-8, divider_pos),
                 arrowprops=dict(arrowstyle="-", color=line_color))
        plt.axhline(y=divider_pos, color=line_color, linestyle='-')

        # Increment n_running_responses
        n_running_responses += n_per_condition

        # Get list of PIDs in this condition
        condition_pids = condition_pid_df[condition_pid_df['condition'] == condition_name]['participant_id'].to_list()
        condition_pid_dict[condition_name] = condition_pids

    # Remove axis tick labels
    plt.xticks([])
    plt.yticks([])

    if save_dir is not None:
        save_path = os.path.join(save_dir, 'participant_correlation.pdf')
        plt.savefig(save_path)

    plt.show()
    plt.clf()

    return condition_pid_dict

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
Functions for visualizing single items
'''
# def prepare_data_individual_items(df,
#                                   groupings,
#                                   significant_items=[]):
#     '''
#     Given df of ratings, return "unpivoted" DF with columns: condition, participant_id, item, and rating

#     Arg(s):
#         df : pd.DataFrame of ratings
#         groupings : dict[str : dict[str : str]]
#             Outer dict is for Weisman/Colombatto, inner is for body/heart/mind/experience/intelligence -> items
#         significant_items : list[str]
#             list of items to mark with *
#     '''
#     # Remove columns weisman_X, and colombatto_X
#     drop_list = ['weisman_', 'colombatto_']
#     cols_to_drop = [col for col in df.columns if any(x in col for x in drop_list)]
#     df = df.drop(columns=cols_to_drop)



#     # Set which columns to "keep" (id_vars) and which to unpivot (value_vars)
#     value_vars = list(df.columns)
#     id_vars = ['condition', 'participant_id']
#     for id_var in id_vars:
#         value_vars.remove(id_var)

#     # Unpivot
#     return_df = df.melt(
#         id_vars=id_vars,
#         value_vars=value_vars,
#         var_name='item',
#         value_name='rating')

#     # Assert number of rows is correct
#     assert len(return_df) == len(df) * 40, "Expected 40 items X {} participants = {} rows, have {} rows in total".format(
#         len(df), len(df) * 40, len(return_df))

#     # Add the colombatto and weisman groupings columns
#     for grouping_source, grouping in groupings.items():
#         group_mapping = {}
#         for category_name, category_mental_states in grouping.items():
#             for mental_state in category_mental_states:
#                 group_mapping[mental_state] = category_name
#         return_df[grouping_source] = return_df['item'].map(group_mapping)

#     # Rename columns with items that are significant
#     if significant_items is not None and len(significant_items) > 0:
#         rename_mapping = {}
#         for item in value_vars:
#             if item in significant_items:
#                 rename_mapping[item] = "{}*".format(item)
#             else:
#                 rename_mapping[item] = item

#         return_df['item'] = return_df['item'].map(rename_mapping)
#         # df = df.rename(columns=rename_mapping)
#     return return_df

# def get_y_order(groupings,
#                 sort_columns,
#                 ascending,
#                 rating_stats_df,
#                 significant_items=[]):
#     '''
#     Sort rows based on sort_columns and return sorted item column

#     Arg(s):
#         groupings : dict[str : dict[str : list[str]]]
#         sort_columns : list[str]
#             which columns to sort by
#         ascending : list[bool]
#             whether each sorting step should be ascending or descending order
#         rating_stats_df : pd.DataFrame
#             DF of mean/std of each item in each condition

#     Returns:
#         list[str] : list of sorted items

#     '''
#     # Remove columns weisman_X, and colombatto_X
#     drop_list = ['weisman_', 'colombatto_']
#     rating_stats_df = rating_stats_df[~rating_stats_df['mental_state'].str.contains('|'.join(drop_list))]
#     assert len(rating_stats_df) == 40

#     # Add the colombatto and weisman groupings
#     for grouping_source, grouping in groupings.items():
#         group_mapping = {}
#         for category_name, category_mental_states in grouping.items():
#             for mental_state in category_mental_states:
#                 group_mapping[mental_state] = category_name
#         rating_stats_df.loc[:, grouping_source] = rating_stats_df.loc[:, 'mental_state'].map(group_mapping)

#     # Sort based on sort_columns
#     rating_stats_df = rating_stats_df.sort_values(by=sort_columns, ascending=ascending)

#     # Get sorted list of items
#     y_order = rating_stats_df['mental_state']
#     if significant_items is not None and len(significant_items) > 0:
#         rename_mapping = {}
#         for item in y_order:
#             if item in significant_items:
#                 rename_mapping[item] = "{}*".format(item)
#             else:
#                 rename_mapping[item] = item

#         y_order = y_order.map(rename_mapping)
#     return y_order


# def plot_individual_items(df,
#                           group_column,
#                           y_order,
#                           conditions,
#                           save_dir=None,
#                           save_ext='pdf'):
#     '''
#     Plots items in separate axes side by side
#     Arg(s):
#         df : pd.DataFrame with columns ['item', 'rating', 'participant_id', 'condition', 'weisman', 'colombatto']
#         group_column : str of which grouping to use
#         y_order : list[str] ordered list for y-axis
#     '''
#     plt.clf()

#     # Set color scheme
#     color_keys = df[group_column].unique()
#     palette = sns.color_palette("Set2", len(color_keys))
#     color_map = dict(zip(color_keys, palette))

#     graph = sns.FacetGrid(
#         df,
#         col='condition',
#         col_order=conditions,
#         col_wrap=4,
#         height=15,
#         aspect=0.35,
#         sharex=True,
#         sharey=False)

#     graph.map_dataframe(
#         sns.pointplot,
#         x="rating",
#         y="item",
#         errorbar="ci",
#         capsize=0.0,
#         hue=group_column,
#         order=y_order,
#         palette=color_map,
#         linestyle='none',
#         dodge=True)

#     # Get figure and axes for titles and labeling
#     axes = graph.axes
#     fig = graph.figure

#     # Set/Unset labeling for specific axes
#     for idx, ax in enumerate(axes):
#         # Remove y-labels
#         if idx > 0:
#             ax.set_yticklabels([])
#         else:
#             ax.tick_params(
#                 axis='y',
#                 labelsize=16
#             )
#             ax.set_ylabel("")

#         # Set axis titles based on condition
#         ax.set_title("{}".format(conditions[idx]), fontsize=20)

#         # Remove xlabel for each graph
#         ax.set_xlabel("")
#         ax.tick_params(axis='x', labelsize=16)
#         ax.set_xticks([1, 2, 3, 4, 5, 6, 7])

#         # Set gridlines
#         axes[idx] = ax.grid(True)

#     # Set global x-axis label
#     fig.supxlabel("Mean Rating", fontsize=20, x=0.6)
#     plt.tight_layout()

#     if save_dir is not None:
#         save_path = os.path.join(save_dir, 'single_items_{}.{}'.format(group_column, save_ext))
#         plt.savefig(save_path)
#         utils.informal_log("Saved single item visualization to {}".format(save_path))

#     plt.show()

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

# def plot_individual_items_single_axis(emmeans_df,

#                                       item_group_dict,
#                                       conditions,
#                                       # Sort ascending (False for horiztonal pointplot, True for vertical)
#                                       ascending=False,
#                                       # Figure parameters
#                                       fig=None,
#                                       ax=None,
#                                       orientation='horizontal',
#                                       marker_size=5,
#                                       alpha=1.0,
#                                       color_idxs=None,
#                                       # Here, x as if was horizontal
#                                       xtick_labels=[],
#                                       xlabel=None,
#                                       font_size_dict={},
#                                       fig_size=None,
#                                       save_path=None,
#                                       show=False):
#     '''
#     Arg(s):
#         emmeans_df : pd.DataFrame from analysis.read_emmeans_marginalized_result() for single items results
#         item_group_dict : dict[str] : str
#             item -> category name dictionary
#         conditions : list[str]
#             list of conditions in order
#     '''

#     pivot_df = _format_and_pivot_emmeans_df(
#         emmeans_df=emmeans_df,
#         target_column='item')
#     # Sort by 1) category and 2) increasing mean-Baseline value
#     pivot_df['category'] = pivot_df['item'].apply(lambda x : item_group_dict[x])
#     pivot_df = pivot_df.sort_values(by=['category', 'mean-Baseline'], ascending=ascending)
#     ytick_labels_list = pivot_df['item'].to_list()

#     means = pivot_df[["mean-{}".format(condition) for condition in conditions]].to_numpy().T
#     errors = pivot_df[["ci_error-{}".format(condition) for condition in conditions]].to_numpy().T

#     if orientation == 'horizontal':
#         ylim = (-1, len(pivot_df))
#         ytick_labels_list = pivot_df['item'].to_list()
#         fig, ax = visualizations.pointplot(
#             fig=fig,
#             ax=ax,
#             means=means,
#             errors=errors,
#             orientation=orientation,
#             labels=conditions,
#             ytick_labels=ytick_labels_list,
#             xtick_labels=xtick_labels,
#             xlabel=xlabel,
#             ylabel=None, #'Mental State Items',
#             title=None, #'Mental State Attributions of LLMs',
#             ylim=ylim,
#             fig_size=fig_size,
#             show_grid=True,
#             alpha=alpha,
#             marker_size=marker_size,
#             color_idxs=color_idxs,
#             font_size_dict=font_size_dict,
#             save_path=None,
#             show=False)
#     elif orientation == 'vertical':
#         xtick_labels_list = pivot_df['item'].to_list()
#         ytick_labels = xtick_labels
#         ylabel = xlabel
#         xlim = (-1, len(pivot_df))
#         fig, ax = visualizations.pointplot(
#             fig=fig,
#             ax=ax,
#             means=means,
#             errors=errors,
#             spacing_multiplier=0.12,
#             orientation=orientation,
#             labels=conditions,
#             xtick_labels=xtick_labels_list,
#             # xtick_label_rotation=70,
#             # xlabel='Mental State Items',
#             ytick_labels=ytick_labels,
#             yticks=ytick_labels,
#             ylabel=ylabel,
#             ylim=(0.5, 7.5),
#             # title='Mental State Attributions of LLMs',
#             xlim=xlim,
#             fig_size=fig_size,
#             show_grid=True,
#             alpha=alpha,
#             marker_size=marker_size,
#             color_idxs=color_idxs,
#             font_size_dict=font_size_dict,
#             save_path=None,
#             show=False)

#         # ax.set_ylim((0.5, 7.5))
#         # ax.set_yticks(range(1, 8))
#         # Rotate x-axis labels
#         for label in ax.get_xticklabels():
#             label.set_rotation(40)  # Rotate labels
#             label.set_horizontalalignment('right')  # Align to the right
#             label.set_transform(label.get_transform() + transforms.ScaledTranslation(20 / 72, 0, fig.dpi_scale_trans))

#             # label.set_position((label.get_position()[0] +5.4, label.get_position()[1]))  # Offset down by 0.1 units
#         plt.tight_layout()
#     else:
#         raise ValueError("Orientation {} not supported".format(orientation))

#     # Color ytick_labels based on category
#     if orientation == 'horizontal':
#         tick_labels = ax.get_yticklabels()
#     else:
#         tick_labels = ax.get_xticklabels()
#     for label in tick_labels:
#         label_text = label.get_text()
#         if item_group_dict[label_text] == "body" or item_group_dict[label_text] == "experience":
#             label.set_color("#D81B60")
#         elif item_group_dict[label_text] == "heart" or item_group_dict[label_text] == "intelligence":
#             label.set_color("#1E88E5")
#         else:
#             label.set_color("#004D40")

#     if save_path is not None:
#         plt.savefig(save_path)
#         utils.informal_log("Saved figure to {}".format(save_path))

#     if show:
#         plt.show()

#     return fig, ax, ytick_labels_list

# def plot_category_items_single_axis(emmeans_df,
#                                     conditions,
#                                     # Figure parameters
#                                     fig=None,
#                                     ax=None,
#                                     alpha=1.0,
#                                     color_idxs=None,
#                                     fig_size=None,
#                                     marker_size=6,
#                                     save_path=None,
#                                     show=False):
#     '''
#     Used for plotting category means of items
#     '''
#     pivot_df = _format_and_pivot_emmeans_df(
#         emmeans_df=emmeans_df,
#         target_column='group'
#     )

#     # Sort by descending so it appears (top) body, heart, mind (bottom)
#     pivot_df = pivot_df.sort_values(by=['group'], ascending=False)

#     means = pivot_df[["mean-{}".format(condition) for condition in conditions]].to_numpy().T
#     errors = pivot_df[["ci_error-{}".format(condition) for condition in conditions]].to_numpy().T


#     # Set y-limit
#     ylim = (-1, len(pivot_df))

#     fig, ax = visualizations.pointplot(
#         fig=fig,
#         ax=ax,
#         means=means,
#         errors=errors,
#         orientation='horizontal',
#         marker_size=marker_size,
#         # labels=conditions,
#         ytick_labels=pivot_df['group'].to_list(),
#         xtick_labels=[i for i in range(1, 8)],
#         xlabel='Rating (1-7)',
#         ylabel='Item Categories',
#         ylim=ylim,
#         # title='Mental State Attributions of LLMs',
#         fig_size=fig_size,
#         show_grid=True,
#         alpha=alpha,
#         color_idxs=color_idxs,
#         save_path=None,
#         show=False)

#     # Color ytick_labels based on category
#     ytick_labels = ax.get_yticklabels()
#     for label in ytick_labels:
#         label_text = label.get_text()
#         if label_text == "body" or label_text == "experience":
#             label.set_color("#D81B60")
#         elif label_text == "heart" or label_text == "intelligence":
#             label.set_color("#1E88E5")
#         else:
#             label.set_color("#004D40")

#     # Save and show
#     if save_path is not None:
#         plt.savefig(save_path)
#         utils.informal_log("Saved figure to {}".format(save_path))

#     if show:
#         plt.show()

#     return fig, ax


# def fa_plot(fa_groupings,
#             r_results_path,
#             plot_type='pointplot',
#             orientation='vertical',
#             graph_save_dir=None,
#             save_ext='pdf',
#             show=True):
#     groups = list(fa_groupings['factor_analysis'].keys())
#     conditions = ['Baseline', 'Mechanistic', 'Functional', 'Intentional']

#     utils.ensure_dir(graph_save_dir)
#     emmeans_graph_data, emmeans_df = read_emmeans_marginalized_result(
#         results_path=r_results_path,
#         grouping_source="factor_analysis",
#         conditions=['Baseline', 'Mechanistic', 'Functional', 'Intentional'],
#         marginalized_var="group",
#         marginalized_var_values=groups,
#         save_dir=graph_save_dir,
#         overwrite=True
#     )
#     if plot_type == 'bargraph':
#         grouped_bar_graphs(
#             groups=groups,
#             grouping_source='{}_components'.format(len(groups)),
#             conditions=['Baseline', 'Mechanistic', 'Functional', 'Intentional'],
#             graph_data=emmeans_graph_data,
#             ci_dim='both',
#             jitter_dim=None,
#             line_start=None,
#             save_dir=graph_save_dir,
#             save_ext='pdf')
#     elif plot_type == 'pointplot':
#         # Data prep copied from analysis.ipynb > Graph from EMMeans Output > Category Level Pointplots (separate by condition)
#         pivot_df = _format_and_pivot_emmeans_df(
#             emmeans_df=emmeans_df,
#             target_column='group'
#         )
#         means = pivot_df[["mean-{}".format(condition) for condition in conditions]].to_numpy().T
#         errors = pivot_df[["ci_error-{}".format(condition) for condition in conditions]].to_numpy().T

#         if orientation == 'horizontal':
#             ytick_labels = [label.capitalize() for label in pivot_df['group'].to_list()]
#             ylabel = 'Item Categories'
#             xtick_labels = [i for i in range(1, 8)]
#             xlabel = 'Rating (1-7)'
#             fig_size = (7, 3)
#         else:
#             xtick_labels = [label.capitalize() for label in pivot_df['group'].to_list()]
#             xlabel = 'Item Categories'
#             ytick_labels = [i for i in range(1, 8)]
#             ylabel = 'Rating (1-7)'
#             fig_size = (3, 4)
#             xlim = (-0.5, 2.5)
#             ylim = None
#         if graph_save_dir is not None:
#             fig_save_path = os.path.join(
#                 graph_save_dir,
#                 'fa_category_graph_{}.{}'.format(orientation, save_ext))
#         else:
#             fig_save_path = None
#         fig, ax = visualizations.pointplot(
#             means=means,
#             errors=errors,
#             orientation=orientation,
#             labels=conditions,
#             ytick_labels=ytick_labels,
#             xtick_labels=xtick_labels,
#             xlabel=xlabel,
#             ylabel=ylabel,
#             ylim=ylim,
#             xlim=xlim,
#             title='Mental State Attributions of LLMs',
#             legend_loc='upper right',
#             fig_size=fig_size,
#             color_idxs=[7, 1, 2, 4], # Copied from analysis.ipynb > Make save_dirs
#             marker_size=10,
#             show_grid=True,
#             save_path=fig_save_path,
#             show=show)
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

# def dv_pointplot(dv_df,
#                  dv_labels,
#                  color_idxs=[7, 1, 2, 4],
#                 #  letter_labels=True,
#                  show=True,
#                  save_dir=None,
#                  save_ext='pdf',
#                  overwrite=True):
#     # if not letter_labels:
#     #     dv_labels = {
#     #         'anthro': 'General\nAnthropomorphism',
#     #         'trust': 'Trust',
#     #         'general': 'General\nAttitude',
#     #         'se_how': 'Self-Efficacy\n(How LLMs Work)\n ',
#     #         'se_use': 'Self-Efficacy\n(How to Use LLMs)',
#     #         'confidence': 'Confidence\nof Responses'
#     #     }
#     # else:
#     #     dv_labels = {
#     #         'anthro': 'General\nAnthropomorphism\n(a)',
#     #         'trust': 'Trust\n\n(b)',
#     #         'general': 'General\nAttitude\n(c)',
#     #         'se_how': 'Self-Efficacy\n(How LLMs Work)\n(d)\n',
#     #         'se_use': 'Self-Efficacy\n(How to Use LLMs)\n(e)',
#     #         'confidence': 'Confidence\nof Responses\n(f)'
#     #     }
#     conditions = ['Baseline', 'Mechanistic', 'Functional', 'Intentional']

#     # Create arrays for means and errors that is shape n_cond X n_dvs
#     means = []
#     errors = []
#     for condition in conditions:
#         condition_means = []
#         condition_errors = []
#         for idx, (dv_name, dv_label) in enumerate(dv_labels.items()):
#             cur_df = dv_df[dv_df['condition'] == condition].loc[:, dv_name]

#             mean = np.nanmean(cur_df.to_numpy())
#             sem = stats.sem(cur_df, axis=None, nan_policy='omit')
#             ci = stats.t.interval(
#                 confidence=0.95,
#                 df=len(cur_df)-1,
#                 loc=mean,
#                 scale=sem)
#             ci_error = (ci[1] - ci[0]) / 2
#             condition_means.append(mean)
#             condition_errors.append(ci_error)

#         means.append(condition_means)
#         errors.append(condition_errors)

#     # Graph
#     fig, ax = visualizations.pointplot(
#         means=means,
#         errors=errors,
#         labels=conditions,
#         show_legend=False,
#         orientation='vertical',
#         color_idxs=color_idxs,
#         xlabel='Belief Measured',
#         xtick_labels=list(dv_labels.values()),
#         xtick_label_rotation=0,
#         ylim=[1, 7],
#         ylabel='Rating (1-7)',
#         title='Mean Ratings of Additional Beliefs Across Conditions',
#         show_grid=True,
#         fig_size=(9, 4),
#         show=False
#     )

#     # Add Legend outside of plot?
#     fig.legend(loc='lower center', bbox_to_anchor=(0.5, -0.1), ncol=4)
#     plt.tight_layout()

#     if save_dir is not None:
#         save_path = os.path.join(save_dir, 'dv_pointplot_data.{}'.format(save_ext))
#         if not os.path.exists(save_path) or overwrite:
#             utils.informal_log("Saving DV graph to {}".format(
#                 save_path
#             ))
#             plt.savefig(save_path, bbox_inches='tight')
#         else:
#             utils.informal_log("DV graph already exists at {} and not overwriting".format(
#                 save_path
#             ))
#     if show:
#         plt.show()

# def dv_bargraph(dv_df,
#             #   plot_type='bargraph',
#               color_idxs=[7, 1, 2, 4],
#               save_dir=None,
#               save_ext='pdf',
#               overwrite=True,
#               one_fig=True):
#     dv_labels = {
#         'anthro': 'General Anthropomorphism\n(a)',
#         'trust': 'Trust\n(b)',
#         'general': 'General Attitude\n(c)',
#         'se_how': 'Self-Efficacy (How LLMs Work)\n(d)',
#         'se_use': 'Self-Efficacy (How to Use LLMs)\n(e)',
#         'confidence': 'Confidence of Responses\n(f)'
#     }
#     conditions = ['Baseline', 'Mechanistic', 'Functional', 'Intentional']

#     if one_fig:
#         n_rows = 2
#         n_cols = 3
#         fig, axes = plt.subplots(n_rows, n_cols, figsize=(8.5, 6.4))

#     for idx, (dv_name, dv_label) in enumerate(dv_labels.items()):
#         if one_fig:
#             ax_row = idx // n_cols
#             ax_col = idx % n_cols
#             ax = axes[ax_row][ax_col]

#         means = []
#         errors = []
#         for condition in conditions:
#             # Select rows for this condition and columns for this DV
#             condition_df = dv_df[dv_df['condition'] == condition].loc[:, dv_name]

#             # Obtain mean and 95% CI
#             mean = np.nanmean(condition_df.to_numpy())
#             sem = stats.sem(condition_df, axis=None, nan_policy='omit')
#             ci = stats.t.interval(
#                 confidence=0.95,
#                 df=len(condition_df)-1,
#                 loc=mean,
#                 scale=sem)
#             ci_error = (ci[1] - ci[0]) / 2

#             means.append([mean])
#             errors.append([ci_error])

#         # Graph

#         if one_fig:
#             # xlabel = 'Condition'
#             xlabel = dv_label
#             ylabel = 'Rating (1-7)' #.format(dv_label)
#             # ylim = [0, np.min([np.max(means) + 1.5, 7])]
#             ylim = [0, 7]
#             title = None
#             fig_size = None
#             show_legend = False
#             show = False
#             if idx == 0:
#                 groups = conditions
#             else:
#                 groups = None
#         else:
#             xlabel = 'Condition'
#             ylabel = '{} (1-7)'.format(dv_label)
#             # ylim = [0, np.min([np.max(means) + 1.5, 7])]
#             ylim = [0, 7]
#             title = '{} Across Conditions'.format(dv_label)
#             show_legend = True
#             fig_size = (4, 4.5)
#             groups = conditions
#             show = True

#         if not one_fig and (save_dir is not None or overwrite):
#             save_path = os.path.join(save_dir, '{}_bargraph.{}'.format(dv_name, save_ext))
#             utils.informal_log("Saving {} graph to {}".format(
#                 dv_name, save_path
#             ))
#         else:
#             save_path = None



#         fig, ax = visualizations.bar_graph(
#             fig=fig,
#             ax=ax,
#             data=means,
#             errors=errors,
#             groups=groups,
#             show_legend=show_legend,
#             legend_loc='upper left',
#             xlabel=xlabel,
#             ylabel=ylabel,
#             axlabel_fontsize=10,
#             ylim=ylim,
#             title=title,
#             alpha=0.75,
#             color_idxs=color_idxs,
#             fig_size=fig_size,
#             save_path=save_path,
#             show=show)
#         # elif plot_type == 'pointplot':
#         #     fig, ax = visualizations.pointplot(
#         #         fig=fig,
#         #         ax=ax,
#         #         means=means,
#         #         errors=errors,
#         #         orientation='vertical',
#         #         labels=groups,
#         #         ylim=ylim[1:],
#         #         xlabel=xlabel,
#         #         ylabel=ylabel,
#         #         show_legend=show_legend,
#         #         color_idxs=color_idxs,
#         #         show_grid=True,
#         #         fig_size=fig_size,
#         #         save_path=save_path,
#         #         show=show

#         #     )
#         # else:
#         #     raise ValueError("plot_type '{}' not supported".format(plot_type))
#         if one_fig:
#             axes[ax_row][ax_col] = ax

#     if one_fig:
#         fig.suptitle(
#             "Responses to Additional Dependent Variables Across Conditions",
#             fontsize=16,
#             x=0.5, y=1.05)
#         fig.legend(
#             loc='lower center',
#             ncol=4,
#             fontsize=12,
#             bbox_to_anchor=(0.5, -0.05))

#         # plt.tight_layout()
#         fig.subplots_adjust(hspace=0.3, wspace=0.25)

#         if save_dir is not None or overwrite:
#             save_path = os.path.join(save_dir, 'dv_bargraph.{}'.format(save_ext))
#             utils.informal_log("Saving DV graph to {}".format(
#                 save_path
#             ))
#             plt.savefig(save_path, bbox_inches='tight')

#         plt.show()


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