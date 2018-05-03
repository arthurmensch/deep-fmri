import os
import re

import nibabel
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from nilearn._utils import check_niimg
from nilearn.image import index_img
from nilearn.input_data import NiftiMasker
from nistats.design_matrix import make_design_matrix
from os.path import expanduser
from os.path import join
from sklearn.utils import gen_batches

# regex for contrasts
CON_REAL_REGX = ("set fmri\(con_real(?P<con_num>\d+?)\.(?P<ev_num>\d+?)\)"
                 " (?P<con_val>\S+)")

# regex for "Number of EVs"
NUM_EV_REGX = """set fmri\(evs_orig\) (?P<evs_orig>\d+)
set fmri\(evs_real\) (?P<evs_real>\d+)
set fmri\(evs_vox\) (?P<evs_vox>\d+)"""

# regex for "Number of contrasts"
NUM_CON_REGX = """set fmri\(ncon_orig\) (?P<ncon>\d+)
set fmri\(ncon_real\) (?P<ncon_real>\d+)"""

# regex for "# EV %i title"
EV_TITLE_REGX = """set fmri\(evtitle\d+?\) \"(?P<evtitle>.+)\""""

# regex for "Title for contrast_real %i"
CON_TITLE_REGX = """set fmri\(conname_real\.\d+?\) \"(?P<conname_real>.+)\""""

# regex for "Basic waveform shape (EV %i)"
# 0 : Square
# 1 : Sinusoid
# 2 : Custom (1 entry per volume)
# 3 : Custom (3 column format)
# 4 : Interaction
# 10 : Empty (all zeros)
EV_SHAPE_REGX = """set fmri\(shape\d+\) (?P<shape>[0|1|3])"""

# regex for "Custom EV file (EV %i)"
EV_CUSTOM_FILE_REGX = """set fmri\(custom\d+?\) \"(?P<custom>.+)\""""

TASK_LIST = ['EMOTION', 'WM', 'MOTOR', 'RELATIONAL',
             'GAMBLING', 'SOCIAL', 'LANGUAGE']

EVS = {'EMOTION': {'EMOTION_Stats.csv',
                   'Sync.txt',
                   'fear.txt',
                   'neut.txt'},
       'GAMBLING': {
           'GAMBLING_Stats.csv',
           'Sync.txt',
           'loss.txt',
           'loss_event.txt',
           'neut_event.txt',
           'win.txt',
           'win_event.txt',
       },
       'LANGUAGE': {
           'LANGUAGE_Stats.csv',
           'Sync.txt',
           'cue.txt',
           'math.txt',
           'present_math.txt',
           'present_story.txt',
           'question_math.txt',
           'question_story.txt',
           'response_math.txt',
           'response_story.txt',
           'story.txt',
       },
       'MOTOR': {
           'Sync.txt',
           'cue.txt',
           'lf.txt',
           'lh.txt',
           'rf.txt',
           'rh.txt',
           't.txt',
       },
       'RELATIONAL': {
           'RELATIONAL_Stats.csv',
           'Sync.txt',
           'error.txt',
           'match.txt',
           'relation.txt',
       },
       'SOCIAL': {
           'SOCIAL_Stats.csv',
           'Sync.txt',
           'mental.txt',
           'mental_resp.txt',
           'other_resp.txt',
           'rnd.txt',
       },
       'WM': {
           '0bk_body.txt',
           '0bk_cor.txt',
           '0bk_err.txt',
           '0bk_faces.txt',
           '0bk_nlr.txt',
           '0bk_places.txt',
           '0bk_tools.txt',
           '2bk_body.txt',
           '2bk_cor.txt',
           '2bk_err.txt',
           '2bk_faces.txt',
           '2bk_nlr.txt',
           '2bk_places.txt',
           '2bk_tools.txt',
           'Sync.txt',
           'WM_Stats.csv',
           'all_bk_cor.txt',
           'all_bk_err.txt'}
       }

CONTRASTS = [["WM", 1, "2BK_BODY"],
             ["WM", 2, "2BK_FACE"],
             ["WM", 3, "2BK_PLACE"],
             ["WM", 4, "2BK_TOOL"],
             ["WM", 5, "0BK_BODY"],
             ["WM", 6, "0BK_FACE"],
             ["WM", 7, "0BK_PLACE"],
             ["WM", 8, "0BK_TOOL"],
             ["WM", 9, "2BK"],
             ["WM", 10, "0BK"],
             ["WM", 11, "2BK-0BK"],
             ["WM", 12, "neg_2BK"],
             ["WM", 13, "neg_0BK"],
             ["WM", 14, "0BK-2BK"],
             ["WM", 15, "BODY"],
             ["WM", 16, "FACE"],
             ["WM", 17, "PLACE"],
             ["WM", 18, "TOOL"],
             ["WM", 19, "BODY-AVG"],
             ["WM", 20, "FACE-AVG"],
             ["WM", 21, "PLACE-AVG"],
             ["WM", 22, "TOOL-AVG"],
             ["WM", 23, "neg_BODY"],
             ["WM", 24, "neg_FACE"],
             ["WM", 25, "neg_PLACE"],
             ["WM", 26, "neg_TOOL"],
             ["WM", 27, "AVG-BODY"],
             ["WM", 28, "AVG-FACE"],
             ["WM", 29, "AVG-PLACE"],
             ["WM", 30, "AVG-TOOL"],
             ["GAMBLING", 1, "PUNISH"],
             ["GAMBLING", 2, "REWARD"],
             ["GAMBLING", 3, "PUNISH-REWARD"],
             ["GAMBLING", 4, "neg_PUNISH"],
             ["GAMBLING", 5, "neg_REWARD"],
             ["GAMBLING", 6, "REWARD-PUNISH"],
             ["MOTOR", 1, "CUE"],
             ["MOTOR", 2, "LF"],
             ["MOTOR", 3, "LH"],
             ["MOTOR", 4, "RF"],
             ["MOTOR", 5, "RH"],
             ["MOTOR", 6, "T"],
             ["MOTOR", 7, "AVG"],
             ["MOTOR", 8, "CUE-AVG"],
             ["MOTOR", 9, "LF-AVG"],
             ["MOTOR", 10, "LH-AVG"],
             ["MOTOR", 11, "RF-AVG"],
             ["MOTOR", 12, "RH-AVG"],
             ["MOTOR", 13, "T-AVG"],
             ["MOTOR", 14, "neg_CUE"],
             ["MOTOR", 15, "neg_LF"],
             ["MOTOR", 16, "neg_LH"],
             ["MOTOR", 17, "neg_RF"],
             ["MOTOR", 18, "neg_RH"],
             ["MOTOR", 19, "neg_T"],
             ["MOTOR", 20, "neg_AVG"],
             ["MOTOR", 21, "AVG-CUE"],
             ["MOTOR", 22, "AVG-LF"],
             ["MOTOR", 23, "AVG-LH"],
             ["MOTOR", 24, "AVG-RF"],
             ["MOTOR", 25, "AVG-RH"],
             ["MOTOR", 26, "AVG-T"],
             ["LANGUAGE", 1, "MATH"],
             ["LANGUAGE", 2, "STORY"],
             ["LANGUAGE", 3, "MATH-STORY"],
             ["LANGUAGE", 4, "STORY-MATH"],
             ["LANGUAGE", 5, "neg_MATH"],
             ["LANGUAGE", 6, "neg_STORY"],
             ["SOCIAL", 1, "RANDOM"],
             ["SOCIAL", 2, "TOM"],
             ["SOCIAL", 3, "RANDOM-TOM"],
             ["SOCIAL", 4, "neg_RANDOM"],
             ["SOCIAL", 5, "neg_TOM"],
             ["SOCIAL", 6, "TOM-RANDOM"],
             ["RELATIONAL", 1, "MATCH"],
             ["RELATIONAL", 2, "REL"],
             ["RELATIONAL", 3, "MATCH-REL"],
             ["RELATIONAL", 4, "REL-MATCH"],
             ["RELATIONAL", 5, "neg_MATCH"],
             ["RELATIONAL", 6, "neg_REL"],
             ["EMOTION", 1, "FACES"],
             ["EMOTION", 2, "SHAPES"],
             ["EMOTION", 3, "FACES-SHAPES"],
             ["EMOTION", 4, "neg_FACES"],
             ["EMOTION", 5, "neg_SHAPES"],
             ["EMOTION", 6, "SHAPES-FACES"]]

DATA_DIR = expanduser('~/data/HCP')


def fetch_hcp_timeseries(data_type='rest'):
    if data_type not in ['task', 'rest']:
        raise ValueError("Wrong data type. Expected 'rest' or 'task', got"
                         "%s" % data_type)

    subjects = ['100307', '178950']

    res = []
    for subject in subjects:
        subject_dir = join(DATA_DIR, str(subject), 'MNINonLinear', 'Results')
        for data_type in ['task', 'rest']:
            if data_type is 'task':
                sessions = TASK_LIST
            else:
                sessions = ['1', '2']
            for session in sessions:
                for direction in ['LR', 'RL']:
                    if data_type == 'task':
                        task = session
                        root_filename = 'tfMRI_%s_%s' % (task, direction)
                    else:
                        root_filename = 'rfMRI_REST%s_%s' % (session,
                                                             direction)
                    root_dir = join(subject_dir, root_filename)
                    filename = join(root_dir, root_filename + '.nii.gz')
                    mask = join(root_dir, root_filename + '_SBRef.nii.gz')
                    confounds = ['Movement_AbsoluteRMS_mean.txt',
                                 'Movement_AbsoluteRMS.txt',
                                 'Movement_Regressors_dt.txt',
                                 'Movement_Regressors.txt',
                                 'Movement_RelativeRMS_mean.txt',
                                 'Movement_RelativeRMS.txt']
                    res_dict = {'filename': filename, 'mask': mask}
                    for i, confound in enumerate(confounds):
                        res_dict['confound_%i' % i] = join(root_dir, confound)
                    if data_type is 'task':
                        feat_file = join(root_dir,
                                         "tfMRI_%s_%s_hp200_s4_level1.fsf"
                                         % (task, direction))
                        res_dict['feat_file'] = feat_file
                        for i, ev in enumerate(EVS[task]):
                            res_dict['ev_%i' % i] = join(root_dir, 'EVs', ev)
                    res_dict['subject'] = subject
                    res_dict['direction'] = direction
                    if data_type == 'rest':
                        res_dict['session'] = 'REST' + session
                    else:
                        res_dict['session'] = task
                    res.append(res_dict)

    res = pd.DataFrame(res)
    res.set_index(['subject', 'session', 'direction'], inplace=True)
    return res


def _get_abspath_relative_to_file(filename, ref_filename):
    """
    Returns the absolute path of a given filename relative to a reference
    filename (ref_filename).

    """

    # we only handle files
    assert os.path.isfile(ref_filename)

    old_cwd = os.getcwd()  # save CWD
    os.chdir(os.path.dirname(ref_filename))  # we're in context now
    abspath = os.path.abspath(filename)  # bing0!
    os.chdir(old_cwd)  # restore CWD

    return abspath


def read_fsl_design_file(design_filename):
    """
    Scrapes an FSL design file for the list of contrasts.

    Returns
    -------
    conditions: list of n_conditions strings
        condition (EV) titles

    timing_files: list of n_condtions strings
        absolute paths of files containing timing info for each condition_id

    contrast_ids: list of n_contrasts strings
        contrast titles

    contrasts: 2D array of shape (n_contrasts, n_conditions)
        array of contrasts, one line per contrast_id; one column per
        condition_id

    Raises
    ------
    AssertionError or IndexError if design_filename is corrupt (not in
    official FSL format)

    """

    # read design file
    design_conf = open(design_filename, 'r').read()

    # scrape n_conditions and n_contrasts
    n_conditions_orig = int(re.search(NUM_EV_REGX,
                                      design_conf).group("evs_orig"))
    n_conditions = int(re.search(NUM_EV_REGX, design_conf).group("evs_real"))
    n_contrasts = int(re.search(NUM_CON_REGX, design_conf).group("ncon_real"))

    # initialize 2D array of contrasts
    contrasts = np.zeros((n_contrasts, n_conditions))

    # lookup EV titles
    conditions = [item.group("evtitle") for item in re.finditer(
        EV_TITLE_REGX, design_conf)]
    assert len(conditions) == n_conditions_orig

    # lookup contrast titles
    contrast_ids = [item.group("conname_real") for item in re.finditer(
        CON_TITLE_REGX, design_conf)]
    assert len(contrast_ids) == n_contrasts

    # lookup EV (condition) custom files
    timing_files = [_get_abspath_relative_to_file(item.group("custom"),
                                                  design_filename)
                    for item in re.finditer(EV_CUSTOM_FILE_REGX, design_conf)]

    # lookup the contrast values
    count = 0
    for item in re.finditer(CON_REAL_REGX, design_conf):
        count += 1
        value = float(item.group('con_val'))

        i = int(item.group('con_num')) - 1
        j = int(item.group('ev_num')) - 1

        # roll-call
        assert 0 <= i < n_contrasts, item.group()
        assert 0 <= j < n_conditions, item.group()

        contrasts[i, j] = value

    # roll-call
    assert count == n_contrasts * n_conditions, count

    return conditions, timing_files, list(zip(contrast_ids, contrasts))


def make_paradigm_from_timing_files(timing_files, trial_types=None):
    if not trial_types is None:
        assert len(trial_types) == len(timing_files)

    onsets = []
    durations = []
    amplitudes = []
    curated_trial_types = []
    count = 0
    for timing_file in timing_files:
        timing = np.loadtxt(timing_file)
        if timing.ndim == 1:
            timing = timing[np.newaxis, :]

        if trial_types is None:
            trial_type = os.path.basename(timing_file).lower(
            ).split('.')[0]
        else:
            trial_type = trial_types[count]
        curated_trial_types += [trial_type] * timing.shape[0]

        count += 1

        if timing.shape[1] == 3:
            onsets += list(timing[..., 0])
            durations += list(timing[..., 1])
            amplitudes += list(timing[..., 2])
        elif timing.shape[1] == 2:
            onsets += list(timing[..., 0])
            durations += list(timing[..., 1])
            amplitudes = durations + list(np.ones(len(timing)))
        elif timing.shape[1] == 1:
            onsets += list(timing[..., 0])
            durations += list(np.zeros(len(timing)))
            amplitudes = durations + list(np.ones(len(timing)))
        else:
            raise TypeError(
                "Timing info must either be 1D array of onsets of 2D "
                "array with 2 or 3 columns: the first column is for "
                "the onsets, the second for the durations, and the "
                "third --if present-- if for the amplitudes; got %s" % timing)

    return pd.DataFrame({'trial_type': curated_trial_types,
                         'onset': onsets,
                         'duration': durations,
                         'modulation': amplitudes})


def main():
    df = fetch_hcp_timeseries()
    for subject, sub_df in df.groupby('subject'):
        # Mask is the same for the same subject
        mask = sub_df.iloc[0]['mask']
        mask = check_niimg(mask)
        data = (mask.get_data() != 0).astype('uint8')
        affine = mask.get_affine()
        del mask
        mask = nibabel.Nifti1Image(data, affine)
        masker = NiftiMasker(smoothing_fwhm=None, mask_img=mask,
                             memory_level=0).fit()
        mask = masker.mask_img_.get_data()
        output_dir = expanduser('~/data/HCP_masked')
        np.save(join(output_dir, '%s_mask' % subject), mask)
        Parallel(n_jobs=15, verbose=10)(delayed(single_unmask)
                                        (index, filename, masker)
                                        for index, filename in
                                        sub_df['filename'].iteritems())
        # paradigm_files = sub_df.iloc[:-4][['filename', 'feat_file']]
        # Parallel(n_jobs=10, verbose=10)(delayed(single_paradigm)
        #                                 (index, design_file)
        #                                 for index, design_file in
        #                                 paradigm_files.iterrows())


def single_unmask(index, filename, masker):
    subject, session, direction = index
    img = check_niimg(filename)
    n_samples = img.shape[3]
    for batch_num, batch in enumerate(gen_batches(n_samples, 300)):
        name = '%s_%s_%s_%s' % (subject, session,
                                direction, batch_num)
        print('Saving', name)
        sub_img = index_img(img, batch)
        img_2d = masker.transform(sub_img).astype('float32')
        img_masked = masker.inverse_transform(img_2d)
        del img_2d
        data = img_masked.get_data() / 16000
        del img_masked
        output_dir = expanduser('~/data/HCP_masked')
        np.save(join(output_dir, name), data)
        del data


def single_paradigm(index, df):
    img, design_file = df.values
    n_scans = check_niimg(img).shape[3]
    slice_time_ref = 0.
    hrf_model = 'glover'
    drift_model = 'cosine'
    period_cut = 128
    drift_order = 1
    fir_delays = [0]
    min_onset = -24
    t_r = 0.8

    subject, session, direction = index
    trial_types, timing_files, contrasts = read_fsl_design_file(
        design_file)

    # fix timing filenames as we load the fsl file one directory
    # higher than expected
    timing_files = [tf.replace("EVs", "tfMRI_%s_%s/EVs" % (
        session, direction)) for tf in timing_files]

    # make design matrix
    events = make_paradigm_from_timing_files(timing_files,
                                             trial_types=trial_types)
    start_time = slice_time_ref * t_r
    end_time = (n_scans - 1 + slice_time_ref) * t_r
    frame_times = np.linspace(start_time, end_time, n_scans)
    design = make_design_matrix(frame_times, events,
                                hrf_model, drift_model,
                                period_cut, drift_order,
                                fir_delays, None, None, min_onset)
    output_dir = expanduser('~/data/HCP_masked')
    for batch_num, batch in enumerate(gen_batches(n_scans, 300)):
        name = '%s_%s_%s_%i_design' % (subject, session, direction,
                                       batch_num)
        np.save(join(output_dir, name), design[batch])


if __name__ == '__main__':
    main()
