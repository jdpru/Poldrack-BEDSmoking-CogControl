from nilearn.glm.first_level import compute_regressor
import numpy as np
import pandas as pd

# ignore get_mean_rt? rt_mns.csv does not exist. 
#ref_rt = 0.8

# def get_mean_rt(task):
#     """
#     Grabs precomputed mean RT from 
#     /oak/stanford/groups/russpold/data/uh2/aim4/analysis_code/utils_lev1/rt_mns.csv
#     """
#     mn_rt_file = ('/oak/stanford/groups/russpold/data/uh2/aim2/analysis_code/'
#         'utils_lev1/rt_mns.csv')
#     mn_rts = pd.read_csv(mn_rt_file)
#     mn_rt_task = mn_rts[task].values[0]
#     return mn_rt_task
    
def get_mean_rt(task, events_df):
    """
    Calculates mean_rt for a given subject. 
    Input: task, events file
    Output: Single mean rt value for that subject-task
    """
    df = events_df
    mean_rt = None
    if task == 'stopSignal':
        df['trial_type'].fillna('n/a', inplace=True)
        subset = df.query("(trial_type.str.contains('go') and response_time >= 0.2 and junk == 0)" +
                          "or (trial_type.str.contains('stop_failure') and response_time >= 0.2 and junk == 0)", engine='python')
        mean_rt = subset['response_time'].mean()
        print("STOP SIGNAL MEAN RT IS:", mean_rt)

        
    else:
        df['trial_type'].fillna('n/a', inplace=True)
        subset = df.query("trial_type != 'n/a' and response_time >= 0.2 and junk == 0")
        mean_rt = subset['response_time'].mean()
        print("MEAN RT IS:", mean_rt)

    return mean_rt
    
    
# def calculate_mean_rt():
#     base_dir = '/oak/stanford/groups/russpold/data/network_grant/discovery_BIDS_21.0.1/derivatives/fitlins_data/*/'
#     tasks = ['cuedTS', 'directedForgetting', 'flanker', 'goNogo',
#                     'nBack', 'stopSignal', 'spatialTS', 'shapeMatching',
#                     'stopSignalWDirectedForgetting', 'stopSignalWFlanker',
#                     'directedForgettingWFlanker']
#     mean_rt_dict = {}
#     for task in tasks:
#         mean_rts = []
#         event_files = glob.glob(base_dir+f'/*/func/*{task}_*events.tsv')
#         if 'stopSignal' in task:
#             for event_file in event_files:
#                 df = pd.read_csv(event_file, sep='\t')
#                 df['trial_type'].fillna('n/a', inplace=True)
#                 subset = df.query("(trial_type.str.contains('go') and response_time >= 0.2 and key_press == correct_response and junk == 0)" +
#                                   "or (trial_type.str.contains('stop_failure') and response_time >= 0.2 and junk == 0)", engine='python')
#                 mean_rt = subset['response_time'].mean()
#                 mean_rts.append(mean_rt)
#             mean_rt_dict[task] = sum(mean_rts)/len(mean_rts)
#         else:
#             for event_file in event_files:
#                 df = pd.read_csv(event_file, sep='\t')
#                 df['trial_type'].fillna('n/a', inplace=True)
#                 subset = df.query("key_press == correct_response and trial_type != 'n/a' and response_time >= 0.2 and junk == 0")
#                 mean_rt = subset['response_time'].mean()
#                 mean_rts.append(mean_rt)
#             mean_rt_dict[task] = sum(mean_rts)/len(mean_rts)
#     return mean_rt_dict

# mean_rt_dict = calculate_mean_rt()


def make_regressor_and_derivative(n_scans, tr, events_df, add_deriv,
                   amplitude_column=None, duration_column=None,
                   onset_column=None, subset=None, demean_amp=False, 
                   cond_id = 'cond'):
    """ Creates regressor and derivative using spm + derivative option in
        nilearn's compute_regressor
        Input:
          n_scans: number of scans
          tr: time resolution in seconds
          events_df: events data frame
          add_deriv: "yes"/"no", whether or not derivatives of regressors should
                     be included
          amplitude_column: Required.  Amplitude column from events_df
          duration_column: Required.  Duration column from events_df
          onset_column: optional.  if not specified "onset" is the default
          subset: optional.  Boolean for subsetting rows of events_df
          demean_amp: Whether amplitude should be mean centered
          cond_id: Name for regressor that is created.  Note "cond_derivative" will
            be assigned as name to the corresponding derivative
        Output:
          regressors: 2 column pandas data frame containing main regressor and derivative
    """
    if subset == None:
        events_df['temp_subset'] = True
        subset = 'temp_subset == True'
    if onset_column == None:
        onset_column = 'onset'
    if amplitude_column == None or duration_column == None:
        print('Must enter amplitude and duration columns')
        return
    if amplitude_column not in events_df.columns:
        print("must specify amplitude column that exists in events_df")
        return
    if duration_column not in events_df.columns:
        print("must specify duration column that exists in events_df")
        return
    reg_3col = events_df.query(subset)[[onset_column, duration_column, amplitude_column]]
    reg_3col = reg_3col.rename(
        columns={duration_column: "duration",
        amplitude_column: "modulation"})
    if demean_amp:
        reg_3col['modulation'] = reg_3col['modulation'] - \
        reg_3col['modulation'].mean()
    if add_deriv == 'deriv_yes':
        hrf_model = 'spm + derivative'
    else:
        hrf_model= 'spm'    
    regressor_array, regressor_names = compute_regressor(
        np.transpose(np.array(reg_3col)),
        hrf_model,
        np.arange(n_scans)*tr+tr/2,
        con_id=cond_id
    ) 
    regressors =  pd.DataFrame(regressor_array, columns=regressor_names) 
    return regressors
    


def define_nuisance_trials(events_df, task):
    """
    Splits junk trials into omission, commission and too_fast, with the exception
    of twoByTwo where too_fast alsoo includes first trial of block
    Note, these categories do not apply to WATT3 or CCTHot
    inputs: 
        events_df: the pandas events data frame
        task: The task name
    output:
        too_fast, omission, commission: indicators for each junk trial type
    """

    if task in ['stopSignal']:
        omission = ((events_df.trial_type == 'go') &
                    (events_df.key_press == -1))
        commission = ((events_df.trial_type == 'go') &
                      (events_df.correct != 1) &
                      (events_df.response_time >= .2))
        too_fast = ((events_df.trial_type == 'go') &
                    (events_df.key_press != -1) &
                    (events_df.response_time < .2))
    elif task in ['motorSelectiveStop']:
        trial_type_list = ['crit_go', 'noncrit_nosignal', 'noncrit_signal']
        omission = ((events_df.trial_type.isin(trial_type_list)) &
                    (events_df.key_press == -1))
        commission = ((events_df.trial_type.isin(trial_type_list)) &
                      (events_df.correct != 1) &
                      (events_df.response_time >= .2))
        too_fast = ((events_df.trial_type.isin(trial_type_list)) &
                    (events_df.key_press != -1) &
                    (events_df.response_time < .2))
    elif task in ['discountFix']:  
        omission = (events_df.key_press == -1)
        commission = 0*omission
        too_fast = (events_df.response_time < .2)
    elif task in ['manipulationTask']:
        trial_type_list = ['present_neutral', 'present_valence', 'future_neutral', 'future_valence']  
        omission = ((events_df.trial_type.isin(trial_type_list)) &
                    (events_df.response == -1))
        commission = 0*omission
        too_fast = (events_df.response_time < .2)
    else:
    # If task is not 'manipulationTask', set 'omission' to 0 for all rows
        events_df['omission'] = 0

    events_df['omission'] = 1 * omission
    events_df['commission'] = 1 * commission
    events_df['too_fast'] = 1 * too_fast
    percent_junk = np.mean(omission | commission | too_fast)
    return events_df, percent_junk


def make_basic_stopsignal_desmat(events_file, add_deriv, 
    regress_rt, n_scans, tr, confound_regressors
):
    """Creates basic stop signal regressors (and derivatives)
       Input
         events_df: events data frame (events.tsv data)
         n_scans: Number of scans
         tr: time resolutionmo
         confound_regressors: Confounds derived from fmriprep output
       Return
          design_matrix: pandas data frame including all regressors (and derivatives)
          contrasts: dictionary of contrasts in nilearn friendly format
          rt_subset: Boolean for extracting correct rows of events_df in the case that 
              rt regressors are requeset.
    """
    events_df = pd.read_csv(events_file, sep = '\t')
    events_df, percent_junk = define_nuisance_trials(events_df, 'stopSignal')
    print(events_df.columns.tolist())
    subset_main_regressors = 'too_fast == 0 and commission == 0 and omission == 0 and onset > 0'
    events_df['constant_1_column'] = 1  
    too_fast_regressor = make_regressor_and_derivative(
            n_scans=n_scans, tr=tr, events_df=events_df, add_deriv = add_deriv,
            amplitude_column="too_fast", duration_column="constant_1_column",
            subset='onset > 0', demean_amp = False, cond_id = 'too_fast'
        )
    omission_regressor = make_regressor_and_derivative(
            n_scans=n_scans, tr=tr, events_df=events_df, add_deriv = add_deriv,
            amplitude_column="omission", duration_column="constant_1_column",
            subset='onset > 0', demean_amp = False, cond_id = 'omission'
        )
    commission_regressor = make_regressor_and_derivative(
            n_scans=n_scans, tr=tr, events_df=events_df, add_deriv = add_deriv,
            amplitude_column="commission", duration_column="constant_1_column",
            subset='onset > 0', demean_amp = False, cond_id = 'commission'
        )

    go = make_regressor_and_derivative(
        n_scans=n_scans, tr=tr, events_df=events_df, add_deriv = add_deriv,
        amplitude_column="constant_1_column", duration_column="constant_1_column",
        subset=  subset_main_regressors + " and trial_type == 'go'", 
        demean_amp=False, cond_id='go'
    )
    stop_success = make_regressor_and_derivative(
        n_scans=n_scans, tr=tr, events_df=events_df, add_deriv = add_deriv,
        amplitude_column="constant_1_column", duration_column="constant_1_column",
        subset=subset_main_regressors + " and trial_type == 'stop_success'", 
        demean_amp=False, cond_id='stop_success'
    )
    stop_failure = make_regressor_and_derivative(
        n_scans=n_scans, tr=tr, events_df=events_df, add_deriv = add_deriv,
        amplitude_column="constant_1_column", duration_column="constant_1_column",
        subset=subset_main_regressors + " and trial_type == 'stop_failure'", 
        demean_amp=False, cond_id='stop_failure'
    )
    design_matrix = pd.concat([go, stop_success, stop_failure, too_fast_regressor, 
        omission_regressor, commission_regressor, confound_regressors], axis=1)
    contrasts = {'go': 'go', 
                    'stop_success': 'stop_success',
                    'stop_failure': 'stop_failure',
                    'stop_success-go': 'stop_success-go',
                    'stop_failure-go': 'stop_failure-go',
                    'stop_success-stop_failure': 'stop_success-stop_failure',
                    'stop_failure-stop_success': 'stop_failure-stop_success',
                    'task': '.333*go + .333*stop_failure + .333*stop_success'
                  }
    if regress_rt == 'rt_centered':
        rt_subset = subset_main_regressors + ' and trial_type != "stop_success"'
        mn_rt = get_mean_rt('stopSignal', events_df)
        events_df['response_time_centered'] = events_df.response_time - mn_rt
        rt = make_regressor_and_derivative(
        n_scans=n_scans, tr=tr, events_df=events_df, add_deriv = add_deriv,
        amplitude_column="response_time_centered", duration_column="constant_1_column",
        subset=rt_subset, demean_amp=False, cond_id='response_time'
        ) 
        design_matrix = pd.concat([design_matrix, rt], axis=1)
        contrasts["response_time"] = "response_time"

    if regress_rt == 'rt_uncentered':
        rt_subset = subset_main_regressors + ' and trial_type != "stop_success"'
        rt = make_regressor_and_derivative(
        n_scans=n_scans, tr=tr, events_df=events_df, add_deriv = add_deriv,
        amplitude_column="response_time", duration_column="constant_1_column",
        subset=rt_subset, demean_amp=False, cond_id='response_time'
        ) 
        design_matrix = pd.concat([design_matrix, rt], axis=1)
        contrasts["response_time"] = "response_time"
    return design_matrix, contrasts, percent_junk, events_df




def make_basic_manipulation_desmat(events_file, add_deriv, 
                                   regress_rt, n_scans, tr, confound_regressors):
    """Creates basic manipulation task regressors (and derivatives)
       Input
         events_df: events data frame (events.tsv data)
         n_scans: Number of scans
         tr: time resolution
         confound_regressors: Confounds derived from fmriprep output
       Return
          design_matrix: pandas data frame including all regressors (and derivatives)
          contrasts: dictionary of contrasts in nilearn friendly format
    """
    events_df = pd.read_csv(events_file, sep='\t')

    events_df, percent_junk = define_nuisance_trials(events_df, 'manipulationTask')
    
    #If the rating omitted (aka qualifies as an omission), this makes sure that the cue and the probe also are marked as omissions (= -1)
    omission_rows = events_df.index[events_df['omission'] == -1].tolist()
    for row in omission_rows:
        events_df.loc[row-2:row-1, 'omission'] = -1

    subset_main_regressors = ('omission == 0')
    events_df['constant_1_column'] = 1

    #Creates an omission regressor
    omission_regressor = make_regressor_and_derivative(
        n_scans=n_scans, tr=tr, events_df=events_df, add_deriv = add_deriv,
        amplitude_column="omission", duration_column="constant_1_column", 
        demean_amp = False, cond_id = 'omission'
    )

    commission_regressor = make_regressor_and_derivative(
        n_scans=n_scans, tr=tr, events_df=events_df, add_deriv = add_deriv,
        amplitude_column="commission", duration_column="constant_1_column",
        subset='onset > 0', demean_amp = False, cond_id = 'commission'
     )

    too_fast_regressor = make_regressor_and_derivative(
        n_scans=n_scans, tr=tr, events_df=events_df, add_deriv = add_deriv,
        amplitude_column="too_fast", duration_column="constant_1_column",
        subset='onset > 0', demean_amp = False, cond_id = 'too_fast'
    )

    # Create stick and parametric cue regressor
    cue_stick_regressor = make_regressor_and_derivative(
        n_scans=n_scans, tr=tr, events_df=events_df,
        onset_column='onset', duration_column='duration', subset=subset_main_regressors + " and duration == 2",
        amplitude_column='constant_1_column', add_deriv=False, 
        cond_id='cue_stick'
    )

    #creating a cue_amplitude column where LATER cues would get an amplitude of 1 and all NOW cues an amplitude of -1 
    events_df['cue_amplitude'] = 0
    events_df.loc[events_df['which_cue'] == 'LATER', 'cue_amplitude'] = 1
    events_df.loc[events_df['which_cue'] == 'NOW', 'cue_amplitude'] = -1

    cue_para_regressor = make_regressor_and_derivative(
        n_scans=n_scans, tr=tr, events_df=events_df,
        onset_column='onset', duration_column='duration', subset=subset_main_regressors + " and duration == 2",
        amplitude_column='cue_amplitude', add_deriv=False, 
        cond_id='cue_parametric'
    )

    # Create stimulus regressors for 'now' and 'later' valence/neutral
    stimulus_regressors = {}
    for condition in ['present_valence', 'present_neutral', 'future_valence', 'future_neutral']:
        subset_condition = f"duration == 3 and trial_type == '{condition}'"
        stimulus_regressors[condition] = make_regressor_and_derivative(
            n_scans=n_scans, tr=tr, events_df=events_df,
            onset_column=f'onset', duration_column='duration', subset=subset_condition,
            amplitude_column='constant_1_column', add_deriv=False,
            cond_id=f'{condition}'
        )
    
    # Create stick and parametric rating regressor
    rating_stick_regressor = make_regressor_and_derivative(
        n_scans=n_scans, tr=tr, events_df=events_df,
        onset_column='onset', duration_column='duration', subset=subset_main_regressors + " and duration == 3",
        amplitude_column='constant_1_column', add_deriv=False,
        cond_id='rating_stick'
    )
    
    #creating a mean centered response column (where the ratings are centred)
    events_df['centered_rating'] = pd.to_numeric(events_df['response'], errors='coerce').fillna(0) - pd.to_numeric(events_df['response'], errors='coerce').mean()

    # Ensure centered_rating is only calculated during appropriate phases
    events_df.loc[events_df['trial_id'].isin(['cue', 'probe']), 'centered_rating'] = None

    rating_para_regressor = make_regressor_and_derivative(
        n_scans=n_scans, tr=tr, events_df=events_df,
        onset_column='onset', duration_column='duration', subset=subset_main_regressors + " and duration == 3",
        amplitude_column='centered_rating', add_deriv=False,
        cond_id='rating_parametric'
    )

    # Combine all the stimulus regressors into a form that can be then combined with the other regressors
    stimulus_reg_concatenated = pd.concat(list(stimulus_regressors.values()), axis=1)

    # Combine all regressors to form design matrix
    design_matrix = pd.concat(
    [omission_regressor, commission_regressor, too_fast_regressor, cue_stick_regressor, 
    stimulus_reg_concatenated, rating_stick_regressor, rating_para_regressor, confound_regressors], axis=1)
 
    contrasts = {'present_valence-present_neutral': 'present_valence-present_neutral', 
                'future_valence-future_neutral': 'future_valence-future_neutral',
                'present_valence-future_valence': 'present_valence-future_valence',
                'present_neutral-future_neutral': 'present_neutral-future_neutral',
                '(future_valence+future_neutral)-(present_valence+present_neutral)': '(future_valence*.5+future_neutral*.5)-(present_valence*.5+present_neutral*.5)',
                '(future_valence+present_valence)-(future_neutral+present_neutral)': '(future_valence*.5+present_valence*.5)-(future_neutral*.5+present_neutral*.5)',
                'task': '.25*present_neutral + .25*present_valence + .25*future_neutral + .25*future_valence'} 

    
    if regress_rt == 'rt_centered':
        subset_main_regressors = ('omission == 0')
        mn_rt = get_mean_rt('manipulationTask', events_df)
        print("rt_centered mn_rt", mn_rt)
        events_df['response_time_centered'] = pd.to_numeric(events_df.response_time, errors='coerce').fillna(0) - mn_rt
        rt = make_regressor_and_derivative(
        n_scans=n_scans, tr=tr, events_df=events_df, add_deriv = add_deriv, 
        subset = subset_main_regressors + " and duration == 3",
        amplitude_column="response_time_centered", duration_column="constant_1_column" ,demean_amp=False, 
        cond_id='response_time'
        ) 
        design_matrix = pd.concat([design_matrix, rt], axis=1)
        contrasts["response_time"] = "response_time"

    # Ensure response_time_centered is only calculated during appropriate phases
        events_df.loc[events_df['trial_id'].isin(['cue', 'probe']), 'response_time_centered'] = None

    if regress_rt == 'rt_uncentered':
        rt = make_regressor_and_derivative(
        n_scans=n_scans, tr=tr, events_df=events_df, add_deriv = add_deriv,
        amplitude_column="response_time", duration_column="constant_1_column", demean_amp=False, 
        cond_id='response_time'
        ) 
        design_matrix = pd.concat([design_matrix, rt], axis=1)
        contrasts["response_time"] = "response_time"
    
    return design_matrix, contrasts, percent_junk, events_df    




def make_basic_discount_fix_desmat(events_file, add_deriv,
    regress_rt, n_scans, tr, confound_regressors
):
    """Creates basic discount fix regressors (and derivatives)
       Input
         events_df: events data frame (events.tsv data)
         n_scans: Number of scans
         tr: time resolution
         confound_regressors: Confounds derived from fmriprep output
       Return
          design_matrix: pandas data frame including all regressors (and derivatives)
          contrasts: dictionary of contrasts in nilearn friendly format
          rt_subset: Boolean for extracting correct rows of events_df in the case that 
              rt regressors are requeset.
    """

    events_df = pd.read_csv(events_file, sep = '\t')
    events_df, percent_junk = define_nuisance_trials(events_df, 'discountFix')
    #commission and omission are all 0s by definition
    subset_main_regressors = ('too_fast == 0 and key_press != -1 and onset > 0')
    events_df['constant_1_column'] = 1  
    events_df['choice_parametric'] = -1
    events_df.loc[events_df.trial_type == 'larger_later',
                  'choice_parametric'] = 1

    too_fast_regressor = make_regressor_and_derivative(
            n_scans=n_scans, tr=tr, events_df=events_df, add_deriv = add_deriv,
            amplitude_column="too_fast", duration_column="constant_1_column",
            subset=None, demean_amp = False, cond_id = 'too_fast'
        )
    task = make_regressor_and_derivative(
        n_scans=n_scans, tr=tr, events_df=events_df, add_deriv = add_deriv,
        amplitude_column="constant_1_column", duration_column="constant_1_column",
        subset='too_fast == 0', demean_amp=False, 
        cond_id='task'
    )
    choice = make_regressor_and_derivative(
        n_scans=n_scans, tr=tr, events_df=events_df, add_deriv = add_deriv,
        amplitude_column="choice_parametric", duration_column="constant_1_column",
        subset='too_fast == 0', demean_amp=True, 
        cond_id='choice'
    )
    design_matrix = pd.concat([task, choice, too_fast_regressor, 
        confound_regressors], axis=1)
    contrasts = {'task': 'task',
                 'choice': 'choice'}
    if regress_rt == 'rt_centered':
        mn_rt = get_mean_rt('discountFix', events_df)
        events_df['response_time_centered'] = events_df.response_time - mn_rt
        rt = make_regressor_and_derivative(
        n_scans=n_scans, tr=tr, events_df=events_df, add_deriv = add_deriv,
        amplitude_column="response_time_centered", duration_column="constant_1_column",
        subset=subset_main_regressors, demean_amp=False, cond_id='response_time'
        ) 
        design_matrix = pd.concat([design_matrix, rt], axis=1)
        contrasts["response_time"] = "response_time"
    if regress_rt == 'rt_uncentered':
        rt = make_regressor_and_derivative(
        n_scans=n_scans, tr=tr, events_df=events_df, add_deriv = add_deriv,
        amplitude_column="response_time", duration_column="constant_1_column",
        subset=subset_main_regressors, demean_amp=False, cond_id='response_time'
        ) 
        design_matrix = pd.concat([design_matrix, rt], axis=1)
        contrasts["response_time"] = "response_time"
    return design_matrix, contrasts, percent_junk, events_df



def make_basic_motor_selective_stop_desmat(events_file, add_deriv,
    regress_rt, n_scans, tr, confound_regressors
    ):
    """Creates basic Motor selective stop regressors (and derivatives)
       Input
         events_df: events data frame (events.tsv data)
         n_scans: Number of scans
         tr: time resolution
         confound_regressors: Confounds derived from fmriprep output
       Return
          design_matrix: pandas data frame including all regressors (and derivatives)
          contrasts: dictionary of contrasts in nilearn friendly format
          rt_subset: Boolean for extracting correct rows of events_df in the case that 
              rt regressors are requeset.
    """
    events_df = pd.read_csv(events_file, sep = '\t')
    events_df, percent_junk = define_nuisance_trials(events_df, 'motorSelectiveStop')
    subset_main_regressors = ('too_fast == 0 and commission == 0 and omission == 0 and key_press != -1')
    #why don't we have key_press != -1 in the StopSignal subset_main_regressors??? 
    events_df['constant_1_column'] = 1  

    too_fast_regressor = make_regressor_and_derivative(
            n_scans=n_scans, tr=tr, events_df=events_df, add_deriv = add_deriv,
            amplitude_column="too_fast", duration_column="constant_1_column",
            subset=None, demean_amp = False, cond_id = 'too_fast'
        )
    commission_regressor = make_regressor_and_derivative(
            n_scans=n_scans, tr=tr, events_df=events_df, add_deriv = add_deriv,
            amplitude_column="commission", duration_column="constant_1_column",
            subset=None, demean_amp = False, cond_id = 'commission'
        )
    omission_regressor = make_regressor_and_derivative(
            n_scans=n_scans, tr=tr, events_df=events_df, add_deriv = add_deriv,
            amplitude_column="omission", duration_column="constant_1_column",
            subset=None, demean_amp = False, cond_id = 'omission'
        )
    crit_go = make_regressor_and_derivative(
        n_scans=n_scans, tr=tr, events_df=events_df, add_deriv = add_deriv,
        amplitude_column="constant_1_column", duration_column="constant_1_column",
        subset=subset_main_regressors + " and trial_type == 'crit_go'", 
        demean_amp=False, cond_id='crit_go'
    )
    crit_stop_success = make_regressor_and_derivative(
        n_scans=n_scans, tr=tr, events_df=events_df, add_deriv = add_deriv,
        amplitude_column="constant_1_column", duration_column="constant_1_column",
        subset= "too_fast == 0 and commission == 0 and omission == 0 and key_press == -1 and trial_type == 'crit_stop_success'",
        #subset=subset_main_regressors + " and trial_type == 'crit_stop_success'",
        demean_amp=False, cond_id='crit_stop_success'
    )
    crit_stop_failure = make_regressor_and_derivative(
        n_scans=n_scans, tr=tr, events_df=events_df, add_deriv = add_deriv,
        amplitude_column="constant_1_column", duration_column="constant_1_column",
        subset=subset_main_regressors + " and trial_type == 'crit_stop_failure'", 
        demean_amp=False, cond_id='crit_stop_failure'
    )
    noncrit_signal = make_regressor_and_derivative(
        n_scans=n_scans, tr=tr, events_df=events_df, add_deriv = add_deriv,
        amplitude_column="constant_1_column", duration_column="constant_1_column",
        subset=subset_main_regressors + " and trial_type == 'noncrit_signal'", 
        demean_amp=False, cond_id='noncrit_signal'
    )
    noncrit_nosignal = make_regressor_and_derivative(
        n_scans=n_scans, tr=tr, events_df=events_df, add_deriv = add_deriv,
        amplitude_column="constant_1_column", duration_column="constant_1_column",
        subset=subset_main_regressors + " and trial_type == 'noncrit_nosignal'", 
        demean_amp=False, cond_id='noncrit_nosignal'
    )
    design_matrix = pd.concat([crit_go, crit_stop_success, crit_stop_failure,
        noncrit_signal, noncrit_nosignal,too_fast_regressor, 
        commission_regressor, omission_regressor, confound_regressors], axis=1)
    contrasts = {'crit_go': 'crit_go',
                 'crit_stop_success': 'crit_stop_success',
                 'crit_stop_failure': 'crit_stop_failure',
                 'noncrit_signal': 'noncrit_signal',
                 'noncrit_nosignal': 'noncrit_nosignal',
                 'crit_stop_success-crit_go': 'crit_stop_success-crit_go', 
                 'crit_stop_failure-crit_go': 'crit_stop_failure-crit_go', 
                 'crit_stop_success-crit_stop_failure': 'crit_stop_success-crit_stop_failure',
                 'crit_go-noncrit_nosignal': 'crit_go-noncrit_nosignal',
                 'noncrit_signal-noncrit_nosignal': 'noncrit_signal-noncrit_nosignal',
                 'crit_stop_success-noncrit_signal': 'crit_stop_success-noncrit_signal',
                 'crit_stop_failure-noncrit_signal': 'crit_stop_failure-noncrit_signal',
                 'task': '.2*crit_go + .2*crit_stop_success +'
                         '.2*crit_stop_failure + .2*noncrit_signal + .2*noncrit_nosignal'
                 }
    if regress_rt == 'rt_centered':
        rt_subset = subset_main_regressors  + 'and trial_type!="crit_stop_success"'
        mn_rt = get_mean_rt('motorSelectiveStop', events_df)
        events_df['response_time_centered'] = events_df.response_time - mn_rt
        rt = make_regressor_and_derivative(
        n_scans=n_scans, tr=tr, events_df=events_df, add_deriv=add_deriv,
        amplitude_column="response_time_centered", duration_column="constant_1_column",
        subset=rt_subset, demean_amp=False, cond_id='response_time'
        ) 
        design_matrix = pd.concat([design_matrix, rt], axis=1)
        contrasts["response_time"] = "response_time"

    if regress_rt == 'rt_uncentered':
        rt_subset = subset_main_regressors  + " and trial_type!='crit_stop_success'"
        rt = make_regressor_and_derivative(
        n_scans=n_scans, tr=tr, events_df=events_df, add_deriv = add_deriv,
        amplitude_column="response_time", duration_column="constant_1_column",
        subset=rt_subset, demean_amp=False, cond_id='response_time'
        ) 
        design_matrix = pd.concat([design_matrix, rt], axis=1)
        contrasts["response_time"] = "response_time"
    return design_matrix, contrasts, percent_junk, events_df

make_task_desmat_fcn_dict = {
        'stopSignal': make_basic_stopsignal_desmat,
        'discountFix': make_basic_discount_fix_desmat,
        'manipulationTask': make_basic_manipulation_desmat,
        'motorSelectiveStop': make_basic_motor_selective_stop_desmat
    }


