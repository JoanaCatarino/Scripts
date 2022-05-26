

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from plot_functions import figure_style

from pupil_size import load_pupil_size_df

date = 'ADD_DATE_TO_DATA_FRAME'

def all_contrasts_by_blocks(pupil_size_df, ):
    pupil_size = pupil_size_df
    # Plot pupil size per contrast for all block types
    pupil_size = pupil_size.reset_index(drop=True)
    dpi = figure_style()
    colors = ['#47BFD1', '#C89AFF', '#FF9561']
    # Contrast = -1
    f, (ax1) = plt.subplots(1, 1, sharey=True, sharex=False, dpi=dpi)
    lineplt = sns.lineplot(x='time', y='baseline_subtracted', hue='probabilityLeft', data=pupil_size[(pupil_size['contrast'] == -1)], legend='full', ci=68, ax=ax1, estimator=np.median, palette = sns.color_palette(colors))
    ax1.set(xlabel='Time relative to StimON (s)', ylabel='Pupil size (%)', title=f' Contrast = -1' '    ' f'{subject}, {date}', ylim=[-25, 25])
    plt.axvline(x = 0, color = 'black', label = 'Stim Onset', linestyle='dashed')
    ax1.legend(loc='center left', bbox_to_anchor=(1, 0.5), frameon=False) # Put a legend to the right of the current axis
    #plt.savefig(join(Fig_path, f'{subject}_{date}_Contrast_neg1.png'))

    # Contrast = -0.25
    f, (ax1) = plt.subplots(1, 1, sharey=True, sharex=False, dpi=dpi)
    lineplt = sns.lineplot(x='time', y='baseline_subtracted', hue='probabilityLeft', data=pupil_size[(pupil_size['contrast'] == -0.25)], legend='full', ci=68, ax=ax1, estimator=np.median, palette = sns.color_palette(colors))
    ax1.set(xlabel='Time relative to StimON (s)', ylabel='Pupil size (%)', title=f' Contrast = -0.25' '    ' f'{subject}, {date}', ylim=[-25, 25])
    plt.axvline(x = 0, color = 'black', label = 'Stim Onset', linestyle='dashed')
    ax1.legend(loc='center left', bbox_to_anchor=(1, 0.5), frameon=False) # Put a legend to the right of the current axis
    #plt.savefig(join(Fig_path, f'{subject}_{date}_Contrast_neg025.png'))

    # Contrast = -0.125
    f, (ax1) = plt.subplots(1, 1, sharey=True, sharex=False, dpi=dpi)
    lineplt = sns.lineplot(x='time', y='baseline_subtracted', hue='probabilityLeft', data=pupil_size[(pupil_size['contrast'] == -0.125)], legend='full', ci=68, ax=ax1, estimator=np.median, palette = sns.color_palette(colors))
    ax1.set(xlabel='Time relative to StimON (s)', ylabel='Pupil size (%)', title=f' Contrast = -0.125' '    ' f'{subject}, {date}', ylim=[-25, 25])
    plt.axvline(x = 0, color = 'black', label = 'Stim Onset', linestyle='dashed')
    ax1.legend(loc='center left', bbox_to_anchor=(1, 0.5), frameon=False) # Put a legend to the right of the current axis
    #plt.savefig(join(Fig_path, f'{subject}_{date}_Contrast_neg0125.png'))

    # Contrast = -0.0625
    f, (ax1) = plt.subplots(1, 1, sharey=True, sharex=False, dpi=dpi)
    lineplt = sns.lineplot(x='time', y='baseline_subtracted', hue='probabilityLeft', data=pupil_size[(pupil_size['contrast'] == -0.0625)], legend='full', ci=68, ax=ax1, estimator=np.median, palette = sns.color_palette(colors))
    ax1.set(xlabel='Time relative to StimON (s)', ylabel='Pupil size (%)', title=f' Contrast = -0.0625' '    ' f'{subject}, {date}', ylim=[-25, 25])
    plt.axvline(x = 0, color = 'black', label = 'Stim Onset', linestyle='dashed')
    ax1.legend(loc='center left', bbox_to_anchor=(1, 0.5), frameon=False) # Put a legend to the right of the current axis
    #plt.savefig(join(Fig_path, f'{subject}_{date}_Contrast_neg00625.png'))

    # Contrast = +0
    f, (ax1) = plt.subplots(1, 1, sharey=True, sharex=False, dpi=dpi)
    lineplt = sns.lineplot(x='time', y='baseline_subtracted', hue='probabilityLeft', data=pupil_size[(pupil_size['contrast'] == 0) & (pupil_size['contrast'] == -0)], legend='full', ci=68, ax=ax1, estimator=np.median, palette = sns.color_palette(colors))
    ax1.set(xlabel='Time relative to StimON (s)', ylabel='Pupil size (%)', title=f' Contrast = 0 right side' '    ' f'{subject}, {date}', ylim=[-25, 25])
    plt.axvline(x = 0, color = 'black', label = 'Stim Onset', linestyle='dashed')
    ax1.legend(loc='center left', bbox_to_anchor=(1, 0.5), frameon=False) # Put a legend to the right of the current axis
    #plt.savefig(join(Fig_path, f'{subject}_{date}_Contrast_0.png'))

    # Contrast = +1
    f, (ax1) = plt.subplots(1, 1, sharey=True, sharex=False, dpi=dpi)
    lineplt = sns.lineplot(x='time', y='baseline_subtracted', hue='probabilityLeft', data=pupil_size[(pupil_size['contrast'] == 1)], legend='full', ci=68, ax=ax1, estimator=np.median, palette = sns.color_palette(colors))
    ax1.set(xlabel='Time relative to StimON (s)', ylabel='Pupil size (%)', title=f' Contrast = 1' '    ' f'{subject}, {date}', ylim=[-25, 25])
    plt.axvline(x = 0, color = 'black', label = 'Stim Onset', linestyle='dashed')
    ax1.legend(loc='center left', bbox_to_anchor=(1, 0.5), frameon=False) # Put a legend to the right of the current axis
    #plt.savefig(join(Fig_path, f'{subject}_{date}_Contrast_pos1.png'))

    # Contrast = +0.25
    f, (ax1) = plt.subplots(1, 1, sharey=True, sharex=False, dpi=dpi)
    lineplt = sns.lineplot(x='time', y='baseline_subtracted', hue='probabilityLeft', data=pupil_size[(pupil_size['contrast'] == 0.25)], legend='full', ci=68, ax=ax1, estimator=np.median, palette = sns.color_palette(colors))
    ax1.set(xlabel='Time relative to StimON (s)', ylabel='Pupil size (%)', title=f' Contrast = 0.25' '    ' f'{subject}, {date}', ylim=[-25, 25])
    plt.axvline(x = 0, color = 'black', label = 'Stim Onset', linestyle='dashed')
    ax1.legend(loc='center left', bbox_to_anchor=(1, 0.5), frameon=False) # Put a legend to the right of the current axis
    #plt.savefig(join(Fig_path, f'{subject}_{date}_Contrast_pos025.png'))

    # Contrast = +0.125
    f, (ax1) = plt.subplots(1, 1, sharey=True, sharex=False, dpi=dpi)
    lineplt = sns.lineplot(x='time', y='baseline_subtracted', hue='probabilityLeft', data=pupil_size[(pupil_size['contrast'] == 0.125)], legend='full', ci=68, ax=ax1, estimator=np.median, palette = sns.color_palette(colors))
    ax1.set(xlabel='Time relative to StimON (s)', ylabel='Pupil size (%)', title=f' Contrast = 0.125' '    ' f'{subject}, {date}', ylim=[-25, 25])
    plt.axvline(x = 0, color = 'black', label = 'Stim Onset', linestyle='dashed')
    ax1.legend(loc='center left', bbox_to_anchor=(1, 0.5), frameon=False) # Put a legend to the right of the current axis
    #plt.savefig(join(Fig_path, f'{subject}_{date}_Contrast_pos0125.png'))

    # Contrast = +0.0625
    f, (ax1) = plt.subplots(1, 1, sharey=True, sharex=False, dpi=dpi)
    lineplt = sns.lineplot(x='time', y='baseline_subtracted', hue='probabilityLeft', data=pupil_size[(pupil_size['contrast'] == 0.0625)], legend='full', ci=68, ax=ax1, estimator=np.median, palette = sns.color_palette(colors))
    ax1.set(xlabel='Time relative to StimON (s)', ylabel='Pupil size (%)', title=f' Contrast = 0.0625' '    ' f'{subject}, {date}', ylim=[-25, 25])
    plt.axvline(x = 0, color = 'black', label = 'Stim Onset', linestyle='dashed')
    ax1.legend(loc='center left', bbox_to_anchor=(1, 0.5), frameon=False) # Put a legend to the right of the current axis
    #plt.savefig(join(Fig_path, f'{subject}_{date}_Contrast_pos00625.png'))

def all_contrasts_per_block_by_stim_side(pupil_size_df):
    # Plot pupil size per block trial with stim appearing on left and right (per constrast)
    pupil_size = pupil_size_df
    pupil_size = pupil_size.reset_index(drop=True)

    dpi = figure_style()
    colors = ['#47BFD1', '#C89AFF']

    # Contrast 1 probability 0.2
    full_contrast_l = pupil_size[((np.abs(pupil_size['contrast']) == 1) & (pupil_size['probabilityLeft'] == 0.2))]
    f, (ax1) = plt.subplots(1, 1, sharey=True, sharex=False, dpi=dpi)
    lineplt = sns.lineplot(x='time', y='baseline_subtracted', hue='Stim_side', data=full_contrast_l, legend='full', ci=68, ax=ax1, estimator=np.median, palette = sns.color_palette(colors))
    ax1.set(xlabel='Time relative to StimON (s)', ylabel='Pupil size (%)', title=f' Contrast = 1 and probability 0.2' '    ' f'{subject}, {date}', ylim=[-25, 25])
    plt.axvline(x = 0, color = 'black', label = 'Stim Onset', linestyle='dashed')
    ax1.legend(loc='center left', bbox_to_anchor=(1, 0.5), frameon=False) # Put a legend to the right of the current axis

    # Contrast 1 probability 0.5
    f, (ax1) = plt.subplots(1, 1, sharey=True, sharex=False, dpi=dpi)
    lineplt = sns.lineplot(x='time', y='baseline_subtracted', hue='Stim_side', data=pupil_size[((pupil_size['contrast'] == -1) & (pupil_size['probabilityLeft'] == 0.5)) | ((pupil_size['contrast'] == 1) & (pupil_size['probabilityLeft'] == 0.5))], legend='full', ci=68, ax=ax1, estimator=np.median, palette = sns.color_palette(colors))
    ax1.set(xlabel='Time relative to StimON (s)', ylabel='Pupil size (%)', title=f' Contrast = 1 and probability 0.5' '    ' f'{subject}, {date}', ylim=[-25, 25])
    plt.axvline(x = 0, color = 'black', label = 'Stim Onset', linestyle='dashed')
    ax1.legend(loc='center left', bbox_to_anchor=(1, 0.5), frameon=False) # Put a legend to the right of the current axis

    # Contrast 1 probability 0.8
    f, (ax1) = plt.subplots(1, 1, sharey=True, sharex=False, dpi=dpi)
    lineplt = sns.lineplot(x='time', y='baseline_subtracted', hue='Stim_side', data=pupil_size[((pupil_size['contrast'] == -1) & (pupil_size['probabilityLeft'] == 0.8)) | ((pupil_size['contrast'] == 1) & (pupil_size['probabilityLeft'] == 0.8))], legend='full', ci=68, ax=ax1, estimator=np.median, palette = sns.color_palette(colors))
    ax1.set(xlabel='Time relative to StimON (s)', ylabel='Pupil size (%)', title=f' Contrast = 1 and probability 0.8' '    ' f'{subject}, {date}', ylim=[-25, 25])
    plt.axvline(x = 0, color = 'black', label = 'Stim Onset', linestyle='dashed')
    ax1.legend(loc='center left', bbox_to_anchor=(1, 0.5), frameon=False) # Put a legend to the right of the current axis

 # -----------------


    # Contrast 0.25 probability 0.2
    f, (ax1) = plt.subplots(1, 1, sharey=True, sharex=False, dpi=dpi)
    lineplt = sns.lineplot(x='time', y='baseline_subtracted', hue='Stim_side', data=pupil_size[((pupil_size['contrast'] == -0.25) & (pupil_size['probabilityLeft'] == 0.2)) | ((pupil_size['contrast'] == 0.25) & (pupil_size['probabilityLeft'] == 0.2))], legend='full', ci=68, ax=ax1, estimator=np.median, palette = sns.color_palette(colors))
    ax1.set(xlabel='Time relative to StimON (s)', ylabel='Pupil size (%)', title=f' Contrast = 0.25 and probability 0.2' '    ' f'{subject}, {date}', ylim=[-25, 25])
    plt.axvline(x = 0, color = 'black', label = 'Stim Onset', linestyle='dashed')
    ax1.legend(loc='center left', bbox_to_anchor=(1, 0.5), frameon=False) # Put a legend to the right of the current axis

    # Contrast 0.25 probability 0.5
    f, (ax1) = plt.subplots(1, 1, sharey=True, sharex=False, dpi=dpi)
    lineplt = sns.lineplot(x='time', y='baseline_subtracted', hue='Stim_side', data=pupil_size[((pupil_size['contrast'] == -0.25) & (pupil_size['probabilityLeft'] == 0.5)) | ((pupil_size['contrast'] == 0.25) & (pupil_size['probabilityLeft'] == 0.5)) ], legend='full', ci=68, ax=ax1, estimator=np.median, palette = sns.color_palette(colors))
    ax1.set(xlabel='Time relative to StimON (s)', ylabel='Pupil size (%)', title=f' Contrast = 0.25 and probability 0.5' '    ' f'{subject}, {date}', ylim=[-25, 25])
    plt.axvline(x = 0, color = 'black', label = 'Stim Onset', linestyle='dashed')
    ax1.legend(loc='center left', bbox_to_anchor=(1, 0.5), frameon=False) # Put a legend to the right of the current axis

    # Contrast 0.25 probability 0.8
    f, (ax1) = plt.subplots(1, 1, sharey=True, sharex=False, dpi=dpi)
    lineplt = sns.lineplot(x='time', y='baseline_subtracted', hue='Stim_side', data=pupil_size[((pupil_size['contrast'] == -0.25) & (pupil_size['probabilityLeft'] == 0.8)) | ((pupil_size['contrast'] == 0.25) & (pupil_size['probabilityLeft'] == 0.8)) ], legend='full', ci=68, ax=ax1, estimator=np.median, palette = sns.color_palette(colors))
    ax1.set(xlabel='Time relative to StimON (s)', ylabel='Pupil size (%)', title=f' Contrast = 0.25 and probability 0.8' '    ' f'{subject}, {date}', ylim=[-25, 25])
    plt.axvline(x = 0, color = 'black', label = 'Stim Onset', linestyle='dashed')
    ax1.legend(loc='center left', bbox_to_anchor=(1, 0.5), frameon=False) # Put a legend to the right of the current axis

# -----------------

    # Contrast 0.125 probability 0.2
    f, (ax1) = plt.subplots(1, 1, sharey=True, sharex=False, dpi=dpi)
    lineplt = sns.lineplot(x='time', y='baseline_subtracted', hue='Stim_side', data=pupil_size[((pupil_size['contrast'] == -0.125) & (pupil_size['probabilityLeft'] == 0.2)) | ((pupil_size['contrast'] == 0.125) & (pupil_size['probabilityLeft'] == 0.2))], legend='full', ci=68, ax=ax1, estimator=np.median, palette = sns.color_palette(colors))
    ax1.set(xlabel='Time relative to StimON (s)', ylabel='Pupil size (%)', title=f' Contrast = 0.125 and probability 0.2' '    ' f'{subject}, {date}', ylim=[-25, 25])
    plt.axvline(x = 0, color = 'black', label = 'Stim Onset', linestyle='dashed')
    ax1.legend(loc='center left', bbox_to_anchor=(1, 0.5), frameon=False) # Put a legend to the right of the current axis

    # Contrast 0.125 probability 0.5
    f, (ax1) = plt.subplots(1, 1, sharey=True, sharex=False, dpi=dpi)
    lineplt = sns.lineplot(x='time', y='baseline_subtracted', hue='Stim_side', data=pupil_size[((pupil_size['contrast'] == -0.125) & (pupil_size['probabilityLeft'] == 0.5)) | ((pupil_size['contrast'] == 0.125) & (pupil_size['probabilityLeft'] == 0.5))], legend='full', ci=68, ax=ax1, estimator=np.median, palette = sns.color_palette(colors))
    ax1.set(xlabel='Time relative to StimON (s)', ylabel='Pupil size (%)', title=f' Contrast = 0.125 and probability 0.5' '    ' f'{subject}, {date}', ylim=[-25, 25])
    plt.axvline(x = 0, color = 'black', label = 'Stim Onset', linestyle='dashed')
    ax1.legend(loc='center left', bbox_to_anchor=(1, 0.5), frameon=False) # Put a legend to the right of the current axis

    # Contrast 0.125 probability 0.8
    f, (ax1) = plt.subplots(1, 1, sharey=True, sharex=False, dpi=dpi)
    lineplt = sns.lineplot(x='time', y='baseline_subtracted', hue='Stim_side', data=pupil_size[((pupil_size['contrast'] == -0.125) & (pupil_size['probabilityLeft'] == 0.8)) | ((pupil_size['contrast'] == 0.125) & (pupil_size['probabilityLeft'] == 0.8))], legend='full', ci=68, ax=ax1, estimator=np.median, palette = sns.color_palette(colors))
    ax1.set(xlabel='Time relative to StimON (s)', ylabel='Pupil size (%)', title=f' Contrast = 0.125 and probability 0.8' '    ' f'{subject}, {date}', ylim=[-25, 25])
    plt.axvline(x = 0, color = 'black', label = 'Stim Onset', linestyle='dashed')
    ax1.legend(loc='center left', bbox_to_anchor=(1, 0.5), frameon=False) # Put a legend to the right of the current axis


# -----------------

    # Contrast 0.0625 probability 0.2
    f, (ax1) = plt.subplots(1, 1, sharey=True, sharex=False, dpi=dpi)
    lineplt = sns.lineplot(x='time', y='baseline_subtracted', hue='Stim_side', data=pupil_size[((pupil_size['contrast'] == -0.0625) & (pupil_size['probabilityLeft'] == 0.2)) | ((pupil_size['contrast'] == 0.0625) & (pupil_size['probabilityLeft'] == 0.2))], legend='full', ci=68, ax=ax1, estimator=np.median, palette = sns.color_palette(colors))
    ax1.set(xlabel='Time relative to StimON (s)', ylabel='Pupil size (%)', title=f' Contrast = 0.0625 and probability 0.2' '    ' f'{subject}, {date}', ylim=[-25, 25])
    plt.axvline(x = 0, color = 'black', label = 'Stim Onset', linestyle='dashed')
    ax1.legend(loc='center left', bbox_to_anchor=(1, 0.5), frameon=False) # Put a legend to the right of the current axis

    # Contrast 0.0625 probability 0.5
    f, (ax1) = plt.subplots(1, 1, sharey=True, sharex=False, dpi=dpi)
    lineplt = sns.lineplot(x='time', y='baseline_subtracted', hue='Stim_side', data=pupil_size[((pupil_size['contrast'] == -0.0625) & (pupil_size['probabilityLeft'] == 0.5)) | ((pupil_size['contrast'] == 0.0625) & (pupil_size['probabilityLeft'] == 0.5))], legend='full', ci=68, ax=ax1, estimator=np.median, palette = sns.color_palette(colors))
    ax1.set(xlabel='Time relative to StimON (s)', ylabel='Pupil size (%)', title=f' Contrast = 0.0625 and probability 0.5' '    ' f'{subject}, {date}', ylim=[-25, 25])
    plt.axvline(x = 0, color = 'black', label = 'Stim Onset', linestyle='dashed')
    ax1.legend(loc='center left', bbox_to_anchor=(1, 0.5), frameon=False) # Put a legend to the right of the current axis

    # Contrast 0.0625 probability 0.8
    f, (ax1) = plt.subplots(1, 1, sharey=True, sharex=False, dpi=dpi)
    lineplt = sns.lineplot(x='time', y='baseline_subtracted', hue='Stim_side', data=pupil_size[((pupil_size['contrast'] == -0.0625) & (pupil_size['probabilityLeft'] == 0.8)) | ((pupil_size['contrast'] == 0.0625) & (pupil_size['probabilityLeft'] == 0.8))], legend='full', ci=68, ax=ax1, estimator=np.median, palette = sns.color_palette(colors))
    ax1.set(xlabel='Time relative to StimON (s)', ylabel='Pupil size (%)', title=f' Contrast = 0.0625 and probability 0.8' '    ' f'{subject}, {date}', ylim=[-25, 25])
    plt.axvline(x = 0, color = 'black', label = 'Stim Onset', linestyle='dashed')
    ax1.legend(loc='center left', bbox_to_anchor=(1, 0.5), frameon=False) # Put a legend to the right of the current axis

 # -----------------

    # Contrast 0 probability 0.2
    f, (ax1) = plt.subplots(1, 1, sharey=True, sharex=False, dpi=dpi)
    lineplt = sns.lineplot(x='time', y='baseline_subtracted', hue='Stim_side', data=pupil_size[((pupil_size['contrast'] == -0) & (pupil_size['probabilityLeft'] == 0.2)) | ((pupil_size['contrast'] == 0) & (pupil_size['probabilityLeft'] == 0.2))], legend='full', ci=68, ax=ax1, estimator=np.median, palette = sns.color_palette(colors))
    ax1.set(xlabel='Time relative to StimON (s)', ylabel='Pupil size (%)', title=f' Contrast = 0 and probability 0.2' '    ' f'{subject}, {date}', ylim=[-25, 25])
    plt.axvline(x = 0, color = 'black', label = 'Stim Onset', linestyle='dashed')
    ax1.legend(loc='center left', bbox_to_anchor=(1, 0.5), frameon=False) # Put a legend to the right of the current axis

    # Contrast 0 probability 0.5
    f, (ax1) = plt.subplots(1, 1, sharey=True, sharex=False, dpi=dpi)
    lineplt = sns.lineplot(x='time', y='baseline_subtracted', hue='Stim_side', data=pupil_size[((pupil_size['contrast'] == -0) & (pupil_size['probabilityLeft'] == 0.5)) | ((pupil_size['contrast'] == 0) & (pupil_size['probabilityLeft'] == 0.5))], legend='full', ci=68, ax=ax1, estimator=np.median, palette = sns.color_palette(colors))
    ax1.set(xlabel='Time relative to StimON (s)', ylabel='Pupil size (%)', title=f' Contrast = 0 and probability 0.5' '    ' f'{subject}, {date}', ylim=[-25, 25])
    plt.axvline(x = 0, color = 'black', label = 'Stim Onset', linestyle='dashed')
    ax1.legend(loc='center left', bbox_to_anchor=(1, 0.5), frameon=False) # Put a legend to the right of the current axis

    # Contrast 0 probability 0.8
    f, (ax1) = plt.subplots(1, 1, sharey=True, sharex=False, dpi=dpi)
    lineplt = sns.lineplot(x='time', y='baseline_subtracted', hue='Stim_side', data=pupil_size[((pupil_size['contrast'] == -0) & (pupil_size['probabilityLeft'] == 0.8)) | ((pupil_size['contrast'] == 0) & (pupil_size['probabilityLeft'] == 0.8))], legend='full', ci=68, ax=ax1, estimator=np.median, palette = sns.color_palette(colors))
    ax1.set(xlabel='Time relative to StimON (s)', ylabel='Pupil size (%)', title=f' Contrast = 0 and probability 0.8' '    ' f'{subject}, {date}', ylim=[-25, 25])
    plt.axvline(x = 0, color = 'black', label = 'Stim Onset', linestyle='dashed')
    ax1.legend(loc='center left', bbox_to_anchor=(1, 0.5), frameon=False) # Put a legend to the right of the current axis

def all_contrasts_all_blocks_correct_error_by_stim_side_figure(pupil_size_df):
    pupil_size = pupil_size_df
    # CORRECT vs INCORRECT TRIALS
    pupil_size = pupil_size.reset_index(drop=True)

    dpi = figure_style()
    colors = ['#47BFD1', '#C89AFF']


    f, ((ax1, ax3, ax5), (ax2, ax4, ax6)) = plt.subplots(2, 3, figsize=(10,4), sharey=False, sharex=False, dpi=800)
    plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.5, hspace=0.6)


    # Contrast 1 probability 0.2 - CORRECT TRIALS
    sns.lineplot(x='time', y='baseline_subtracted', hue='Stim_side', data=pupil_size[((pupil_size['contrast'] == -1) & (pupil_size['probabilityLeft'] == 0.2) & (pupil_size['Feedback_type'] == 1)) | ((pupil_size['contrast'] == 1) & (pupil_size['probabilityLeft'] == 0.2) & (pupil_size['Feedback_type'] == 1))], legend=None, ci=68, ax=ax1, estimator=np.median, palette = sns.color_palette(colors))
    ax1.set(xlabel='Time relative to StimON (s)', ylabel='Pupil size (%)', title=f' Probability 0.2 - CORRECT', ylim=[-25, 25])
    ax1.plot([0, 0], ax1.get_ylim(), ls='--', color='black', label='Stim Onset')

    # Contrast 1 probability 0.2 - INCORRECT TRIALS
    sns.lineplot(x='time', y='baseline_subtracted', hue='Stim_side', data=pupil_size[((pupil_size['contrast'] == -1) & (pupil_size['probabilityLeft'] == 0.2) & (pupil_size['Feedback_type'] == -1)) | ((pupil_size['contrast'] == 1) & (pupil_size['probabilityLeft'] == 0.2) & (pupil_size['Feedback_type'] == -1))], legend=None, ci=68, ax=ax2, estimator=np.median, palette = sns.color_palette(colors))
    ax2.set(xlabel='Time relative to StimON (s)', ylabel='Pupil size (%)', title=f' Probability 0.2 - INCORRECT', ylim=[-25, 25])
    ax2.plot([0, 0], ax2.get_ylim(), ls='--', color='black', label='Stim Onset')

    # Contrast 1 probability 0.5 - CORRECT TRIALS
    sns.lineplot(x='time', y='baseline_subtracted', hue='Stim_side', data=pupil_size[((pupil_size['contrast'] == -1) & (pupil_size['probabilityLeft'] == 0.5) & (pupil_size['Feedback_type'] == 1)) | ((pupil_size['contrast'] == 1) & (pupil_size['probabilityLeft'] == 0.5) & (pupil_size['Feedback_type'] == 1))], legend=None, ci=68, ax=ax3, estimator=np.median, palette = sns.color_palette(colors))
    ax3.set(xlabel='Time relative to StimON (s)', ylabel='', title=f' Probability 0.5 - CORRECT', ylim=[-25, 25])
    ax3.plot([0, 0], ax3.get_ylim(), ls='--', color='black', label='Stim Onset')

    # Contrast 1 probability 0.5 - INCORRECT TRIALS
    sns.lineplot(x='time', y='baseline_subtracted', hue='Stim_side', data=pupil_size[((pupil_size['contrast'] == -1) & (pupil_size['probabilityLeft'] == 0.5) & (pupil_size['Feedback_type'] == -1)) | ((pupil_size['contrast'] == 1) & (pupil_size['probabilityLeft'] == 0.5) & (pupil_size['Feedback_type'] == -1))], legend=None, ci=68, ax=ax4, estimator=np.median, palette = sns.color_palette(colors))
    ax4.set(xlabel='Time relative to StimON (s)', ylabel='', title=f' Probability 0.5 - INCORRECT', ylim=[-25, 25])
    ax4.plot([0, 0], ax4.get_ylim(), ls='--', color='black', label='Stim Onset')

    # Contrast 1 probability 0.8 - CORRECT TRIALS
    sns.lineplot(x='time', y='baseline_subtracted', hue='Stim_side', data=pupil_size[((pupil_size['contrast'] == -1) & (pupil_size['probabilityLeft'] == 0.8) & (pupil_size['Feedback_type'] == 1)) | ((pupil_size['contrast'] == 1) & (pupil_size['probabilityLeft'] == 0.8) & (pupil_size['Feedback_type'] == 1))], legend='full', ci=68, ax=ax5, estimator=np.median, palette = sns.color_palette(colors))
    ax5.set(xlabel='Time relative to StimON (s)', ylabel='', title=f' Probability 0.8 - CORRECT', ylim=[-25, 25])
    ax5.plot([0, 0], ax5.get_ylim(), ls='--', color='black', label='Stim Onset')
    ax5.legend(loc='center left', bbox_to_anchor=(1, 0.5), frameon=False) # Put a legend to the right of the current axis

    # Contrast 1 probability 0.8 - INCORRECT TRIALS
    sns.lineplot(x='time', y='baseline_subtracted', hue='Stim_side', data=pupil_size[((pupil_size['contrast'] == -1) & (pupil_size['probabilityLeft'] == 0.8) & (pupil_size['Feedback_type'] == -1)) | ((pupil_size['contrast'] == 1) & (pupil_size['probabilityLeft'] == 0.8) & (pupil_size['Feedback_type'] == -1))], legend=None, ci=68, ax=ax6, estimator=np.median, palette = sns.color_palette(colors))
    ax6.set(xlabel='Time relative to StimON (s)', ylabel='', title=f' Probability 0.8 - INCORRECT', ylim=[-25, 25])
    ax6.plot([0, 0], ax6.get_ylim(), ls='--', color='black', label='Stim Onset')


 # ----------

    f, ((ax1, ax3, ax5), (ax2, ax4, ax6)) = plt.subplots(2, 3, figsize=(10, 4), sharey=False, sharex=False, dpi=800)
    plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.5, hspace=0.6)


    # Contrast 0.25 probability 0.2 - CORRECT TRIALS
    sns.lineplot(x='time', y='baseline_subtracted', hue='Stim_side', data=pupil_size[((pupil_size['contrast'] == -0.25) & (pupil_size['probabilityLeft'] == 0.2) & (pupil_size['Feedback_type'] == 1)) | ((pupil_size['contrast'] == 0.25) & (pupil_size['probabilityLeft'] == 0.2) & (pupil_size['Feedback_type'] == 1))], legend=None, ci=68, ax=ax1, estimator=np.median, palette = sns.color_palette(colors))
    ax1.set(xlabel='Time relative to StimON (s)', ylabel='Pupil size (%)', title=f' Probability 0.2 - CORRECT', ylim=[-25, 25])
    ax1.plot([0, 0], ax1.get_ylim(), ls='--', color='black', label='Stim Onset')

    # Contrast 0.25 probability 0.2 - INCORRECT TRIALS
    sns.lineplot(x='time', y='baseline_subtracted', hue='Stim_side', data=pupil_size[((pupil_size['contrast'] == -0.25) & (pupil_size['probabilityLeft'] == 0.2) & (pupil_size['Feedback_type'] == -1)) | ((pupil_size['contrast'] == 0.25) & (pupil_size['probabilityLeft'] == 0.2) & (pupil_size['Feedback_type'] == -1))], legend=None, ci=68, ax=ax2, estimator=np.median, palette = sns.color_palette(colors))
    ax2.set(xlabel='Time relative to StimON (s)', ylabel='Pupil size (%)', title=f' Probability 0.2 - INCORRECT', ylim=[-25, 25])
    ax2.plot([0, 0], ax2.get_ylim(), ls='--', color='black', label='Stim Onset')

    # Contrast 0.25 probability 0.5 - CORRECT TRIALS
    sns.lineplot(x='time', y='baseline_subtracted', hue='Stim_side', data=pupil_size[((pupil_size['contrast'] == -0.25) & (pupil_size['probabilityLeft'] == 0.5) & (pupil_size['Feedback_type'] == 1)) | ((pupil_size['contrast'] == 0.25) & (pupil_size['probabilityLeft'] == 0.5) & (pupil_size['Feedback_type'] == 1))], legend=None, ci=68, ax=ax3, estimator=np.median, palette = sns.color_palette(colors))
    ax3.set(xlabel='Time relative to StimON (s)', ylabel='', title=f' Probability 0.5 - CORRECT', ylim=[-25, 25])
    ax3.plot([0, 0], ax3.get_ylim(), ls='--', color='black', label='Stim Onset')

    # Contrast 0.25 probability 0.5 - INCORRECT TRIALS
    sns.lineplot(x='time', y='baseline_subtracted', hue='Stim_side', data=pupil_size[((pupil_size['contrast'] == -0.25) & (pupil_size['probabilityLeft'] == 0.5) & (pupil_size['Feedback_type'] == -1)) | ((pupil_size['contrast'] == 0.25) & (pupil_size['probabilityLeft'] == 0.5) & (pupil_size['Feedback_type'] == -1))], legend=None, ci=68, ax=ax4, estimator=np.median, palette = sns.color_palette(colors))
    ax4.set(xlabel='Time relative to StimON (s)', ylabel='', title=f' Probability 0.5 - INCORRECT', ylim=[-25, 25])
    ax4.plot([0, 0], ax4.get_ylim(), ls='--', color='black', label='Stim Onset')

    # Contrast 0.25 probability 0.8 - CORRECT TRIALS
    sns.lineplot(x='time', y='baseline_subtracted', hue='Stim_side', data=pupil_size[((pupil_size['contrast'] == -0.25) & (pupil_size['probabilityLeft'] == 0.8) & (pupil_size['Feedback_type'] == 1)) | ((pupil_size['contrast'] == 0.25) & (pupil_size['probabilityLeft'] == 0.8) & (pupil_size['Feedback_type'] == 1))], legend='full', ci=68, ax=ax5, estimator=np.median, palette = sns.color_palette(colors))
    ax5.set(xlabel='Time relative to StimON (s)', ylabel='', title=f' Probability 0.8 - CORRECT', ylim=[-25, 25])
    ax5.plot([0, 0], ax5.get_ylim(), ls='--', color='black', label='Stim Onset')
    ax5.legend(loc='center left', bbox_to_anchor=(1, 0.5), frameon=False) # Put a legend to the right of the current axis

    # Contrast 0.25 probability 0.8 - INCORRECT TRIALS
    sns.lineplot(x='time', y='baseline_subtracted', hue='Stim_side', data=pupil_size[((pupil_size['contrast'] == -0.25) & (pupil_size['probabilityLeft'] == 0.8) & (pupil_size['Feedback_type'] == -1)) | ((pupil_size['contrast'] == 0.25) & (pupil_size['probabilityLeft'] == 0.8) & (pupil_size['Feedback_type'] == -1))], legend=None, ci=68, ax=ax6, estimator=np.median, palette = sns.color_palette(colors))
    ax6.set(xlabel='Time relative to StimON (s)', ylabel='', title=f' Probability 0.8 - INCORRECT', ylim=[-25, 25])
    ax6.plot([0, 0], ax6.get_ylim(), ls='--', color='black', label='Stim Onset')


# ------------

    f, ((ax1, ax3, ax5), (ax2, ax4, ax6)) = plt.subplots(2, 3, figsize=(10, 4), sharey=False, sharex=False, dpi=800)
    plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.5, hspace=0.6)


    # Contrast 0.125 probability 0.2 - CORRECT TRIALS
    sns.lineplot(x='time', y='baseline_subtracted', hue='Stim_side', data=pupil_size[((pupil_size['contrast'] == -0.125) & (pupil_size['probabilityLeft'] == 0.2) & (pupil_size['Feedback_type'] == 1)) | ((pupil_size['contrast'] == 0.125) & (pupil_size['probabilityLeft'] == 0.2) & (pupil_size['Feedback_type'] == 1))], legend=None, ci=68, ax=ax1, estimator=np.median, palette = sns.color_palette(colors))
    ax1.set(xlabel='Time relative to StimON (s)', ylabel='Pupil size (%)', title=f' Probability 0.2 - CORRECT', ylim=[-25, 25])
    ax1.plot([0, 0], ax1.get_ylim(), ls='--', color='black', label='Stim Onset')

    # Contrast 0.125 probability 0.2 - INCORRECT TRIALS
    sns.lineplot(x='time', y='baseline_subtracted', hue='Stim_side', data=pupil_size[((pupil_size['contrast'] == -0.125) & (pupil_size['probabilityLeft'] == 0.2) & (pupil_size['Feedback_type'] == -1)) | ((pupil_size['contrast'] == 0.125) & (pupil_size['probabilityLeft'] == 0.2) & (pupil_size['Feedback_type'] == -1))], legend=None, ci=68, ax=ax2, estimator=np.median, palette = sns.color_palette(colors))
    ax2.set(xlabel='Time relative to StimON (s)', ylabel='Pupil size (%)', title=f' Probability 0.2 - INCORRECT', ylim=[-25, 25])
    ax2.plot([0, 0], ax2.get_ylim(), ls='--', color='black', label='Stim Onset')

    # Contrast 0.125 probability 0.5 - CORRECT TRIALS
    sns.lineplot(x='time', y='baseline_subtracted', hue='Stim_side', data=pupil_size[((pupil_size['contrast'] == -0.125) & (pupil_size['probabilityLeft'] == 0.5) & (pupil_size['Feedback_type'] == 1)) | ((pupil_size['contrast'] == 0.125) & (pupil_size['probabilityLeft'] == 0.5) & (pupil_size['Feedback_type'] == 1))], legend=None, ci=68, ax=ax3, estimator=np.median, palette = sns.color_palette(colors))
    ax3.set(xlabel='Time relative to StimON (s)', ylabel='', title=f' Probability 0.5 - CORRECT', ylim=[-25, 25])
    ax3.plot([0, 0], ax3.get_ylim(), ls='--', color='black', label='Stim Onset')

    # Contrast 0.125 probability 0.5 - INCORRECT TRIALS
    sns.lineplot(x='time', y='baseline_subtracted', hue='Stim_side', data=pupil_size[((pupil_size['contrast'] == -0.125) & (pupil_size['probabilityLeft'] == 0.5) & (pupil_size['Feedback_type'] == -1)) | ((pupil_size['contrast'] == 0.125) & (pupil_size['probabilityLeft'] == 0.5) & (pupil_size['Feedback_type'] == -1))], legend=None, ci=68, ax=ax4, estimator=np.median, palette = sns.color_palette(colors))
    ax4.set(xlabel='Time relative to StimON (s)', ylabel='', title=f' Probability 0.5 - INCORRECT', ylim=[-25, 25])
    ax4.plot([0, 0], ax4.get_ylim(), ls='--', color='black', label='Stim Onset')

    # Contrast 0.125 probability 0.8 - CORRECT TRIALS
    sns.lineplot(x='time', y='baseline_subtracted', hue='Stim_side', data=pupil_size[((pupil_size['contrast'] == -0.125) & (pupil_size['probabilityLeft'] == 0.8) & (pupil_size['Feedback_type'] == 1)) | ((pupil_size['contrast'] == 0.125) & (pupil_size['probabilityLeft'] == 0.8) & (pupil_size['Feedback_type'] == 1))], legend='full', ci=68, ax=ax5, estimator=np.median, palette = sns.color_palette(colors))
    ax5.set(xlabel='Time relative to StimON (s)', ylabel='', title=f' Probability 0.8 - CORRECT', ylim=[-25, 25])
    ax5.plot([0, 0], ax5.get_ylim(), ls='--', color='black', label='Stim Onset')
    ax5.legend(loc='center left', bbox_to_anchor=(1, 0.5), frameon=False) # Put a legend to the right of the current axis

    # Contrast 0.125 probability 0.8 - INCORRECT TRIALS
    sns.lineplot(x='time', y='baseline_subtracted', hue='Stim_side', data=pupil_size[((pupil_size['contrast'] == -0.125) & (pupil_size['probabilityLeft'] == 0.8) & (pupil_size['Feedback_type'] == -1)) | ((pupil_size['contrast'] == 0.125) & (pupil_size['probabilityLeft'] == 0.8) & (pupil_size['Feedback_type'] == -1))], legend=None, ci=68, ax=ax6, estimator=np.median, palette = sns.color_palette(colors))
    ax6.set(xlabel='Time relative to StimON (s)', ylabel='', title=f' Probability 0.8 - INCORRECT', ylim=[-25, 25])
    ax6.plot([0, 0], ax6.get_ylim(), ls='--', color='black', label='Stim Onset')

# ------------

    f, ((ax1, ax3, ax5), (ax2, ax4, ax6)) = plt.subplots(2, 3, figsize=(10, 4), sharey=False, sharex=False, dpi=800)
    plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.5, hspace=0.6)


    # Contrast 0.0625 probability 0.2 - CORRECT TRIALS
    sns.lineplot(x='time', y='baseline_subtracted', hue='Stim_side', data=pupil_size[((pupil_size['contrast'] == -0.0625) & (pupil_size['probabilityLeft'] == 0.2) & (pupil_size['Feedback_type'] == 1)) | ((pupil_size['contrast'] == 0.0625) & (pupil_size['probabilityLeft'] == 0.2) & (pupil_size['Feedback_type'] == 1))], legend=None, ci=68, ax=ax1, estimator=np.median, palette = sns.color_palette(colors))
    ax1.set(xlabel='Time relative to StimON (s)', ylabel='Pupil size (%)', title=f' Probability 0.2 - CORRECT', ylim=[-25, 25])
    ax1.plot([0, 0], ax1.get_ylim(), ls='--', color='black', label='Stim Onset')

    # Contrast 0.0625 probability 0.2 - INCORRECT TRIALS
    sns.lineplot(x='time', y='baseline_subtracted', hue='Stim_side', data=pupil_size[((pupil_size['contrast'] == -0.0625) & (pupil_size['probabilityLeft'] == 0.2) & (pupil_size['Feedback_type'] == -1)) | ((pupil_size['contrast'] == 0.0625) & (pupil_size['probabilityLeft'] == 0.2) & (pupil_size['Feedback_type'] == -1))], legend=None, ci=68, ax=ax2, estimator=np.median, palette = sns.color_palette(colors))
    ax2.set(xlabel='Time relative to StimON (s)', ylabel='Pupil size (%)', title=f' Probability 0.2 - INCORRECT', ylim=[-25, 25])
    ax2.plot([0, 0], ax2.get_ylim(), ls='--', color='black', label='Stim Onset')

    # Contrast 0.0625 probability 0.5 - CORRECT TRIALS
    sns.lineplot(x='time', y='baseline_subtracted', hue='Stim_side', data=pupil_size[((pupil_size['contrast'] == -0.0625) & (pupil_size['probabilityLeft'] == 0.5) & (pupil_size['Feedback_type'] == 1)) | ((pupil_size['contrast'] == 0.0625) & (pupil_size['probabilityLeft'] == 0.5) & (pupil_size['Feedback_type'] == 1))], legend=None, ci=68, ax=ax3, estimator=np.median, palette = sns.color_palette(colors))
    ax3.set(xlabel='Time relative to StimON (s)', ylabel='', title=f' Probability 0.5 - CORRECT', ylim=[-25, 25])
    ax3.plot([0, 0], ax3.get_ylim(), ls='--', color='black', label='Stim Onset')

    # Contrast 0.0625 probability 0.5 - INCORRECT TRIALS
    sns.lineplot(x='time', y='baseline_subtracted', hue='Stim_side', data=pupil_size[((pupil_size['contrast'] == -0.0625) & (pupil_size['probabilityLeft'] == 0.5) & (pupil_size['Feedback_type'] == -1)) | ((pupil_size['contrast'] == 0.0625) & (pupil_size['probabilityLeft'] == 0.5) & (pupil_size['Feedback_type'] == -1))], legend=None, ci=68, ax=ax4, estimator=np.median, palette = sns.color_palette(colors))
    ax4.set(xlabel='Time relative to StimON (s)', ylabel='', title=f' Probability 0.5 - INCORRECT', ylim=[-25, 25])
    ax4.plot([0, 0], ax4.get_ylim(), ls='--', color='black', label='Stim Onset')

    # Contrast 0.0625 probability 0.8 - CORRECT TRIALS
    sns.lineplot(x='time', y='baseline_subtracted', hue='Stim_side', data=pupil_size[((pupil_size['contrast'] == -0.0625) & (pupil_size['probabilityLeft'] == 0.8) & (pupil_size['Feedback_type'] == 1)) | ((pupil_size['contrast'] == 0.0625) & (pupil_size['probabilityLeft'] == 0.8) & (pupil_size['Feedback_type'] == 1))], legend='full', ci=68, ax=ax5, estimator=np.median, palette = sns.color_palette(colors))
    ax5.set(xlabel='Time relative to StimON (s)', ylabel='', title=f' Probability 0.8 - CORRECT', ylim=[-25, 25])
    ax5.plot([0, 0], ax5.get_ylim(), ls='--', color='black', label='Stim Onset')
    ax5.legend(loc='center left', bbox_to_anchor=(1, 0.5), frameon=False) # Put a legend to the right of the current axis

    # Contrast 0.0625 probability 0.8 - INCORRECT TRIALS
    sns.lineplot(x='time', y='baseline_subtracted', hue='Stim_side', data=pupil_size[((pupil_size['contrast'] == -0.0625) & (pupil_size['probabilityLeft'] == 0.8) & (pupil_size['Feedback_type'] == -1)) | ((pupil_size['contrast'] == 0.0625) & (pupil_size['probabilityLeft'] == 0.8) & (pupil_size['Feedback_type'] == -1))], legend=None, ci=68, ax=ax6, estimator=np.median, palette = sns.color_palette(colors))
    ax6.set(xlabel='Time relative to StimON (s)', ylabel='', title=f' Probability 0.8 - INCORRECT', ylim=[-25, 25])
    ax6.plot([0, 0], ax6.get_ylim(), ls='--', color='black', label='Stim Onset')

# ------------

    f, ((ax1, ax3, ax5), (ax2, ax4, ax6)) = plt.subplots(2, 3, figsize=(10, 4), sharey=False, sharex=False, dpi=800)
    plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.5, hspace=0.6)


    # Contrast 0 probability 0.2 - CORRECT TRIALS
    sns.lineplot(x='time', y='baseline_subtracted', hue='Stim_side', data=pupil_size[((pupil_size['contrast'] == -0) & (pupil_size['probabilityLeft'] == 0.2) & (pupil_size['Feedback_type'] == 1)) | ((pupil_size['contrast'] == 0) & (pupil_size['probabilityLeft'] == 0.2) & (pupil_size['Feedback_type'] == 1))], legend=None, ci=68, ax=ax1, estimator=np.median, palette = sns.color_palette(colors))
    ax1.set(xlabel='Time relative to StimON (s)', ylabel='Pupil size (%)', title=f' Probability 0.2 - CORRECT', ylim=[-25, 25])
    ax1.plot([0, 0], ax1.get_ylim(), ls='--', color='black', label='Stim Onset')

    # Contrast 0 probability 0.2 - INCORRECT TRIALS
    sns.lineplot(x='time', y='baseline_subtracted', hue='Stim_side', data=pupil_size[((pupil_size['contrast'] == -0) & (pupil_size['probabilityLeft'] == 0.2) & (pupil_size['Feedback_type'] == -1)) | ((pupil_size['contrast'] == 0) & (pupil_size['probabilityLeft'] == 0.2) & (pupil_size['Feedback_type'] == -1))], legend=None, ci=68, ax=ax2, estimator=np.median, palette = sns.color_palette(colors))
    ax2.set(xlabel='Time relative to StimON (s)', ylabel='Pupil size (%)', title=f' Probability 0.2 - INCORRECT', ylim=[-25, 25])
    ax2.plot([0, 0], ax2.get_ylim(), ls='--', color='black', label='Stim Onset')

    # Contrast 0 probability 0.5 - CORRECT TRIALS
    sns.lineplot(x='time', y='baseline_subtracted', hue='Stim_side', data=pupil_size[((pupil_size['contrast'] == -0) & (pupil_size['probabilityLeft'] == 0.5) & (pupil_size['Feedback_type'] == 1)) | ((pupil_size['contrast'] == 0) & (pupil_size['probabilityLeft'] == 0.5) & (pupil_size['Feedback_type'] == 1))], legend=None, ci=68, ax=ax3, estimator=np.median, palette = sns.color_palette(colors))
    ax3.set(xlabel='Time relative to StimON (s)', ylabel='', title=f' Probability 0.5 - CORRECT', ylim=[-25, 25])
    ax3.plot([0, 0], ax3.get_ylim(), ls='--', color='black', label='Stim Onset')

    # Contrast 0 probability 0.5 - INCORRECT TRIALS
    sns.lineplot(x='time', y='baseline_subtracted', hue='Stim_side', data=pupil_size[((pupil_size['contrast'] == -0) & (pupil_size['probabilityLeft'] == 0.5) & (pupil_size['Feedback_type'] == -1)) | ((pupil_size['contrast'] == 0) & (pupil_size['probabilityLeft'] == 0.5) & (pupil_size['Feedback_type'] == -1))], legend=None, ci=68, ax=ax4, estimator=np.median, palette = sns.color_palette(colors))
    ax4.set(xlabel='Time relative to StimON (s)', ylabel='', title=f' Probability 0.5 - INCORRECT', ylim=[-25, 25])
    ax4.plot([0, 0], ax4.get_ylim(), ls='--', color='black', label='Stim Onset')

    # Contrast 0 probability 0.8 - CORRECT TRIALS
    sns.lineplot(x='time', y='baseline_subtracted', hue='Stim_side', data=pupil_size[((pupil_size['contrast'] == -0) & (pupil_size['probabilityLeft'] == 0.8) & (pupil_size['Feedback_type'] == 1)) | ((pupil_size['contrast'] == 0) & (pupil_size['probabilityLeft'] == 0.8) & (pupil_size['Feedback_type'] == 1))], legend='full', ci=68, ax=ax5, estimator=np.median, palette = sns.color_palette(colors))
    ax5.set(xlabel='Time relative to StimON (s)', ylabel='', title=f' Probability 0.8 - CORRECT', ylim=[-25, 25])
    ax5.plot([0, 0], ax5.get_ylim(), ls='--', color='black', label='Stim Onset')
    ax5.legend(loc='center left', bbox_to_anchor=(1, 0.5), frameon=False) # Put a legend to the right of the current axis

    # Contrast 0 probability 0.8 - INCORRECT TRIALS
    sns.lineplot(x='time', y='baseline_subtracted', hue='Stim_side', data=pupil_size[((pupil_size['contrast'] == -0) & (pupil_size['probabilityLeft'] == 0.8) & (pupil_size['Feedback_type'] == -1)) | ((pupil_size['contrast'] == 0) & (pupil_size['probabilityLeft'] == 0.8) & (pupil_size['Feedback_type'] == -1))], legend=None, ci=68, ax=ax6, estimator=np.median, palette = sns.color_palette(colors))
    ax6.set(xlabel='Time relative to StimON (s)', ylabel='', title=f' Probability 0.8 - INCORRECT', ylim=[-25, 25])
    ax6.plot([0, 0], ax6.get_ylim(), ls='--', color='black', label='Stim Onset')


if __name__ == '__main__':
    subject = "ZFM-02368"
    pupil_size_df = load_pupil_size_df(subject)
    all_contrasts_by_blocks(pupil_size_df)
    all_contrasts_per_block_by_stim_side(pupil_size_df)
    all_contrasts_all_blocks_correct_error_by_stim_side_figure(pupil_size_df)
