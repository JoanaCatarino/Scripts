
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from plot_functions import figure_style
from one.api import ONE
one = ONE()



#pupil_size_df = pd.read_csv('/home/joana/Desktop/data_Pupil_Project/Aligned to stimON/All_animals_Mainen_mean_stimON.csv')
pupil_size_df = pd.read_csv('/home/joana/Desktop/data_Pupil_Project/Aligned to feedback times/All_animals_Mainen_mean_feedbacktimes.csv')
subject = pupil_size_df.subject.unique()


def all_contrasts_by_blocks(pupil_size_df, subject):
    
    pupil_size = pupil_size_df
    
    # Plot pupil size per contrast for all block types
    pupil_size = pupil_size.reset_index(drop=True)
    dpi = figure_style()
    colors = ['#47BFD1', '#C89AFF', '#FF9561']
    
    # Things to change depending on the type of plot: 'legend' and 'label'!

    # Contrast = -1
    f, (ax1) = plt.subplots(1, 1, sharey=True, sharex=False, dpi=dpi)
    lineplt = sns.lineplot(x='time', y='baseline_subtracted', hue='probabilityLeft', data=pupil_size[(pupil_size['contrast'] == -1)], legend='full', ci=68, ax=ax1, estimator=np.median, palette = sns.color_palette(colors))
    ax1.set(xlabel='Time relative to Feedback Times (s)', ylabel='Pupil size (%)', title=f' Contrast = -1' '    ' 'n=20 mice', ylim=[-15, 15])
    plt.axvline(x = 0, color = 'black', label = 'Feedback Times', linestyle='dashed')
    ax1.legend(loc='center left', bbox_to_anchor=(1, 0.5), frameon=False) # P-10, 10ut a legend to the right of the current axis
    sns.despine(trim=True)

    # Contrast = -0.25
    f, (ax1) = plt.subplots(1, 1, sharey=True, sharex=False, dpi=dpi)
    lineplt = sns.lineplot(x='time', y='baseline_subtracted', hue='probabilityLeft', data=pupil_size[(pupil_size['contrast'] == -0.25)], legend='full', ci=68, ax=ax1, estimator=np.median, palette = sns.color_palette(colors))
    ax1.set(xlabel='Time relative to Feedback Times (s)', ylabel='Pupil size (%)', title=f' Contrast = -0.25' '    ' 'n=20 mice', ylim=[-15, 15])
    plt.axvline(x = 0, color = 'black', label = 'Feedback Times', linestyle='dashed')
    ax1.legend(loc='center left', bbox_to_anchor=(1, 0.5), frameon=False) # Put a legend to the right of the current axis
    sns.despine(trim=True)

    # Contrast = -0.125
    f, (ax1) = plt.subplots(1, 1, sharey=True, sharex=False, dpi=dpi)
    lineplt = sns.lineplot(x='time', y='baseline_subtracted', hue='probabilityLeft', data=pupil_size[(pupil_size['contrast'] == -0.125)], legend='full', ci=68, ax=ax1, estimator=np.median, palette = sns.color_palette(colors))
    ax1.set(xlabel='Time relative to Feedback Times (s)', ylabel='Pupil size (%)', title=f' Contrast = -0.125' '    ' 'n=20 mice', ylim=[-15, 15])
    plt.axvline(x = 0, color = 'black', label = 'Feedback Times', linestyle='dashed')
    ax1.legend(loc='center left', bbox_to_anchor=(1, 0.5), frameon=False) # Put a legend to the right of the current axis
    sns.despine(trim=True)

    # Contrast = -0.0625
    f, (ax1) = plt.subplots(1, 1, sharey=True, sharex=False, dpi=dpi)
    lineplt = sns.lineplot(x='time', y='baseline_subtracted', hue='probabilityLeft', data=pupil_size[(pupil_size['contrast'] == -0.0625)], legend='full', ci=68, ax=ax1, estimator=np.median, palette = sns.color_palette(colors))
    ax1.set(xlabel='Time relative to Feedback Times (s)', ylabel='Pupil size (%)', title=f' Contrast = -0.0625' '    ' 'n=20 mice', ylim=[-15, 15])
    plt.axvline(x = 0, color = 'black', label = 'Feedback Times', linestyle='dashed')
    ax1.legend(loc='center left', bbox_to_anchor=(1, 0.5), frameon=False) # Put a legend to the right of the current axis
    sns.despine(trim=True)

    # Contrast = +0
    f, (ax1) = plt.subplots(1, 1, sharey=True, sharex=False, dpi=dpi)
    lineplt = sns.lineplot(x='time', y='baseline_subtracted', hue='probabilityLeft', data=pupil_size[(pupil_size['contrast'] == 0) & (pupil_size['contrast'] == -0)], legend='full', ci=68, ax=ax1, estimator=np.median, palette = sns.color_palette(colors))
    ax1.set(xlabel='Time relative to Feedback Times (s)', ylabel='Pupil size (%)', title=f' Contrast = 0 right side' '    ' 'n=20 mice', ylim=[-15, 15])
    plt.axvline(x = 0, color = 'black', label = 'Feedback Times', linestyle='dashed')
    ax1.legend(loc='center left', bbox_to_anchor=(1, 0.5), frameon=False) # Put a legend to the right of the current axis
    sns.despine(trim=True)

    # Contrast = +1
    f, (ax1) = plt.subplots(1, 1, sharey=True, sharex=False, dpi=dpi)
    lineplt = sns.lineplot(x='time', y='baseline_subtracted', hue='probabilityLeft', data=pupil_size[(pupil_size['contrast'] == 1)], legend='full', ci=68, ax=ax1, estimator=np.median, palette = sns.color_palette(colors))
    ax1.set(xlabel='Time relative to Feedback Times (s)', ylabel='Pupil size (%)', title=f' Contrast = 1' '    ' 'n=20 mice', ylim=[-15, 15])
    plt.axvline(x = 0, color = 'black', label = 'Feedback Times', linestyle='dashed')
    ax1.legend(loc='center left', bbox_to_anchor=(1, 0.5), frameon=False) # Put a legend to the right of the current axis
    sns.despine(trim=True)

    # Contrast = +0.25
    f, (ax1) = plt.subplots(1, 1, sharey=True, sharex=False, dpi=dpi)
    lineplt = sns.lineplot(x='time', y='baseline_subtracted', hue='probabilityLeft', data=pupil_size[(pupil_size['contrast'] == 0.25)], legend='full', ci=68, ax=ax1, estimator=np.median, palette = sns.color_palette(colors))
    ax1.set(xlabel='Time relative to Feedback Times (s)', ylabel='Pupil size (%)', title=f' Contrast = 0.25' '    ' 'n=20 mice', ylim=[-15, 15])
    plt.axvline(x = 0, color = 'black', label = 'Feedback Times', linestyle='dashed')
    ax1.legend(loc='center left', bbox_to_anchor=(1, 0.5), frameon=False) # Put a legend to the right of the current axis
    sns.despine(trim=True)

    # Contrast = +0.125
    f, (ax1) = plt.subplots(1, 1, sharey=True, sharex=False, dpi=dpi)
    lineplt = sns.lineplot(x='time', y='baseline_subtracted', hue='probabilityLeft', data=pupil_size[(pupil_size['contrast'] == 0.125)], legend='full', ci=68, ax=ax1, estimator=np.median, palette = sns.color_palette(colors))
    ax1.set(xlabel='Time relative to Feedback Times (s)', ylabel='Pupil size (%)', title=f' Contrast = 0.125' '    ' 'n=20 mice', ylim=[-15, 15])
    plt.axvline(x = 0, color = 'black', label = 'Feedback Times', linestyle='dashed')
    ax1.legend(loc='center left', bbox_to_anchor=(1, 0.5), frameon=False) # Put a legend to the right of the current axis
    sns.despine(trim=True)

    # Contrast = +0.0625
    f, (ax1) = plt.subplots(1, 1, sharey=True, sharex=False, dpi=dpi)
    lineplt = sns.lineplot(x='time', y='baseline_subtracted', hue='probabilityLeft', data=pupil_size[(pupil_size['contrast'] == 0.0625)], legend='full', ci=68, ax=ax1, estimator=np.median, palette = sns.color_palette(colors))
    ax1.set(xlabel='Time relative to Feedback Times (s)', ylabel='Pupil size (%)', title=f' Contrast = 0.0625' '    ' 'n=20 mice', ylim=[-15, 15])
    plt.axvline(x = 0, color = 'black', label = 'Feedback Times', linestyle='dashed')
    ax1.legend(loc='center left', bbox_to_anchor=(1, 0.5), frameon=False) # Put a legend to the right of the current axis
    sns.despine(trim=True)
    


def all_contrasts_per_block_by_stim_side(pupil_size_df, subject):
    
    # Plot pupil size per block trial with stim appearing on left and right (per constrast)
    pupil_size = pupil_size_df
    pupil_size = pupil_size.reset_index(drop=True)

    dpi = figure_style()
    colors = ['#47BFD1', '#C89AFF']

    # Contrast 1 probability 0.2
    f, (ax1) = plt.subplots(1, 1, sharey=True, sharex=False, dpi=dpi)
    lineplt = sns.lineplot(x='time', y='baseline_subtracted', hue='Stim_side', data=pupil_size[((pupil_size['contrast'] == -1) & (pupil_size['probabilityLeft'] == 0.2)) | ((pupil_size['contrast'] == 1) & (pupil_size['probabilityLeft'] == 0.2))], legend='full', ci=68, ax=ax1, estimator=np.median, palette = sns.color_palette(colors))
    ax1.set(xlabel='Time relative to Feedback Times (s)', ylabel='Pupil size (%)', title=f' Contrast = 1 and probability 0.2' '    ' 'n=20 mice', ylim=[-15, 15])
    plt.axvline(x = 0, color = 'black', label = 'Feedback Times', linestyle='dashed')
    ax1.legend(loc='center left', bbox_to_anchor=(1, 0.5), frameon=False)
    sns.despine(trim=True)
    
    # Contrast 1 probability 0.5
    f, (ax1) = plt.subplots(1, 1, sharey=True, sharex=False, dpi=dpi)
    lineplt = sns.lineplot(x='time', y='baseline_subtracted', hue='Stim_side', data=pupil_size[((pupil_size['contrast'] == -1) & (pupil_size['probabilityLeft'] == 0.5)) | ((pupil_size['contrast'] == 1) & (pupil_size['probabilityLeft'] == 0.5))], legend='full', ci=68, ax=ax1, estimator=np.median, palette = sns.color_palette(colors))
    ax1.set(xlabel='Time relative to Feedback Times (s)', ylabel='Pupil size (%)', title=f' Contrast = 1 and probability 0.5' '    ' 'n=20 mice', ylim=[-15, 15])
    plt.axvline(x = 0, color = 'black', label = 'Feedback Times', linestyle='dashed')
    ax1.legend(loc='center left', bbox_to_anchor=(1, 0.5), frameon=False)
    sns.despine(trim=True)

    # Contrast 1 probability 0.8
    f, (ax1) = plt.subplots(1, 1, sharey=True, sharex=False, dpi=dpi)
    lineplt = sns.lineplot(x='time', y='baseline_subtracted', hue='Stim_side', data=pupil_size[((pupil_size['contrast'] == -1) & (pupil_size['probabilityLeft'] == 0.8)) | ((pupil_size['contrast'] == 1) & (pupil_size['probabilityLeft'] == 0.8))], legend='full', ci=68, ax=ax1, estimator=np.median, palette = sns.color_palette(colors))
    ax1.set(xlabel='Time relative to Feedback Times (s)', ylabel='Pupil size (%)', title=f' Contrast = 1 and probability 0.8' '    ' 'n=20 mice', ylim=[-15, 15])
    plt.axvline(x = 0, color = 'black', label = 'Feedback Times', linestyle='dashed')
    ax1.legend(loc='center left', bbox_to_anchor=(1, 0.5), frameon=False)
    sns.despine(trim=True)

 # -----------------


    # Contrast 0.25 probability 0.2
    f, (ax1) = plt.subplots(1, 1, sharey=True, sharex=False, dpi=dpi)
    lineplt = sns.lineplot(x='time', y='baseline_subtracted', hue='Stim_side', data=pupil_size[((pupil_size['contrast'] == -0.25) & (pupil_size['probabilityLeft'] == 0.2)) | ((pupil_size['contrast'] == 0.25) & (pupil_size['probabilityLeft'] == 0.2))], legend='full', ci=68, ax=ax1, estimator=np.median, palette = sns.color_palette(colors))
    ax1.set(xlabel='Time relative to Feedback Times (s)', ylabel='Pupil size (%)', title=f' Contrast = 0.25 and probability 0.2' '    ' 'n=20 mice', ylim=[-15, 15])
    plt.axvline(x = 0, color = 'black', label = 'Feedback Times', linestyle='dashed')
    ax1.legend(loc='center left', bbox_to_anchor=(1, 0.5), frameon=False)
    sns.despine(trim=True)

    # Contrast 0.25 probability 0.5
    f, (ax1) = plt.subplots(1, 1, sharey=True, sharex=False, dpi=dpi)
    lineplt = sns.lineplot(x='time', y='baseline_subtracted', hue='Stim_side', data=pupil_size[((pupil_size['contrast'] == -0.25) & (pupil_size['probabilityLeft'] == 0.5)) | ((pupil_size['contrast'] == 0.25) & (pupil_size['probabilityLeft'] == 0.5)) ], legend='full', ci=68, ax=ax1, estimator=np.median, palette = sns.color_palette(colors))
    ax1.set(xlabel='Time relative to Feedback Times (s)', ylabel='Pupil size (%)', title=f' Contrast = 0.25 and probability 0.5' '    ' 'n=20 mice', ylim=[-15, 15])
    plt.axvline(x = 0, color = 'black', label = 'Feedback Times', linestyle='dashed')
    ax1.legend(loc='center left', bbox_to_anchor=(1, 0.5), frameon=False)
    sns.despine(trim=True)

    # Contrast 0.25 probability 0.8
    f, (ax1) = plt.subplots(1, 1, sharey=True, sharex=False, dpi=dpi)
    lineplt = sns.lineplot(x='time', y='baseline_subtracted', hue='Stim_side', data=pupil_size[((pupil_size['contrast'] == -0.25) & (pupil_size['probabilityLeft'] == 0.8)) | ((pupil_size['contrast'] == 0.25) & (pupil_size['probabilityLeft'] == 0.8)) ], legend='full', ci=68, ax=ax1, estimator=np.median, palette = sns.color_palette(colors))
    ax1.set(xlabel='Time relative to Feedback Times (s)', ylabel='Pupil size (%)', title=f' Contrast = 0.25 and probability 0.8' '    ' 'n=20 mice', ylim=[-15, 15])
    plt.axvline(x = 0, color = 'black', label = 'Feedback Times', linestyle='dashed')
    ax1.legend(loc='center left', bbox_to_anchor=(1, 0.5), frameon=False)
    sns.despine(trim=True)

# -----------------

    # Contrast 0.125 probability 0.2
    f, (ax1) = plt.subplots(1, 1, sharey=True, sharex=False, dpi=dpi)
    lineplt = sns.lineplot(x='time', y='baseline_subtracted', hue='Stim_side', data=pupil_size[((pupil_size['contrast'] == -0.125) & (pupil_size['probabilityLeft'] == 0.2)) | ((pupil_size['contrast'] == 0.125) & (pupil_size['probabilityLeft'] == 0.2))], legend='full', ci=68, ax=ax1, estimator=np.median, palette = sns.color_palette(colors))
    ax1.set(xlabel='Time relative to Feedback Times (s)', ylabel='Pupil size (%)', title=f' Contrast = 0.125 and probability 0.2' '    ' 'n=20 mice', ylim=[-15, 15])
    plt.axvline(x = 0, color = 'black', label = 'Feedback Times', linestyle='dashed')
    ax1.legend(loc='center left', bbox_to_anchor=(1, 0.5), frameon=False)
    sns.despine(trim=True)

    # Contrast 0.125 probability 0.5
    f, (ax1) = plt.subplots(1, 1, sharey=True, sharex=False, dpi=dpi)
    lineplt = sns.lineplot(x='time', y='baseline_subtracted', hue='Stim_side', data=pupil_size[((pupil_size['contrast'] == -0.125) & (pupil_size['probabilityLeft'] == 0.5)) | ((pupil_size['contrast'] == 0.125) & (pupil_size['probabilityLeft'] == 0.5))], legend='full', ci=68, ax=ax1, estimator=np.median, palette = sns.color_palette(colors))
    ax1.set(xlabel='Time relative to Feedback Times (s)', ylabel='Pupil size (%)', title=f' Contrast = 0.125 and probability 0.5' '    ' 'n=20 mice', ylim=[-15, 15])
    plt.axvline(x = 0, color = 'black', label = 'Feedback Times', linestyle='dashed')
    ax1.legend(loc='center left', bbox_to_anchor=(1, 0.5), frameon=False) 
    sns.despine(trim=True)

    # Contrast 0.125 probability 0.8
    f, (ax1) = plt.subplots(1, 1, sharey=True, sharex=False, dpi=dpi)
    lineplt = sns.lineplot(x='time', y='baseline_subtracted', hue='Stim_side', data=pupil_size[((pupil_size['contrast'] == -0.125) & (pupil_size['probabilityLeft'] == 0.8)) | ((pupil_size['contrast'] == 0.125) & (pupil_size['probabilityLeft'] == 0.8))], legend='full', ci=68, ax=ax1, estimator=np.median, palette = sns.color_palette(colors))
    ax1.set(xlabel='Time relative to Feedback Times (s)', ylabel='Pupil size (%)', title=f' Contrast = 0.125 and probability 0.8' '    ' 'n=20 mice', ylim=[-15, 15])
    plt.axvline(x = 0, color = 'black', label = 'Feedback Times', linestyle='dashed')
    ax1.legend(loc='center left', bbox_to_anchor=(1, 0.5), frameon=False) 
    sns.despine(trim=True)


# -----------------

    # Contrast 0.0625 probability 0.2
    f, (ax1) = plt.subplots(1, 1, sharey=True, sharex=False, dpi=dpi)
    lineplt = sns.lineplot(x='time', y='baseline_subtracted', hue='Stim_side', data=pupil_size[((pupil_size['contrast'] == -0.0625) & (pupil_size['probabilityLeft'] == 0.2)) | ((pupil_size['contrast'] == 0.0625) & (pupil_size['probabilityLeft'] == 0.2))], legend='full', ci=68, ax=ax1, estimator=np.median, palette = sns.color_palette(colors))
    ax1.set(xlabel='Time relative to Feedback Times (s)', ylabel='Pupil size (%)', title=f' Contrast = 0.0625 and probability 0.2' '    ' 'n=20 mice', ylim=[-15, 15])
    plt.axvline(x = 0, color = 'black', label = 'Feedback Times', linestyle='dashed')
    ax1.legend(loc='center left', bbox_to_anchor=(1, 0.5), frameon=False)
    sns.despine(trim=True)

    # Contrast 0.0625 probability 0.5
    f, (ax1) = plt.subplots(1, 1, sharey=True, sharex=False, dpi=dpi)
    lineplt = sns.lineplot(x='time', y='baseline_subtracted', hue='Stim_side', data=pupil_size[((pupil_size['contrast'] == -0.0625) & (pupil_size['probabilityLeft'] == 0.5)) | ((pupil_size['contrast'] == 0.0625) & (pupil_size['probabilityLeft'] == 0.5))], legend='full', ci=68, ax=ax1, estimator=np.median, palette = sns.color_palette(colors))
    ax1.set(xlabel='Time relative to Feedback Times (s)', ylabel='Pupil size (%)', title=f' Contrast = 0.0625 and probability 0.5' '    ' 'n=20 mice', ylim=[-15, 15])
    plt.axvline(x = 0, color = 'black', label = 'Feedback Times', linestyle='dashed')
    ax1.legend(loc='center left', bbox_to_anchor=(1, 0.5), frameon=False) 
    sns.despine(trim=True)

    # Contrast 0.0625 probability 0.8
    f, (ax1) = plt.subplots(1, 1, sharey=True, sharex=False, dpi=dpi)
    lineplt = sns.lineplot(x='time', y='baseline_subtracted', hue='Stim_side', data=pupil_size[((pupil_size['contrast'] == -0.0625) & (pupil_size['probabilityLeft'] == 0.8)) | ((pupil_size['contrast'] == 0.0625) & (pupil_size['probabilityLeft'] == 0.8))], legend='full', ci=68, ax=ax1, estimator=np.median, palette = sns.color_palette(colors))
    ax1.set(xlabel='Time relative to Feedback Times (s)', ylabel='Pupil size (%)', title=f' Contrast = 0.0625 and probability 0.8' '    ' 'n=20 mice', ylim=[-15, 15])
    plt.axvline(x = 0, color = 'black', label = 'Feedback Times', linestyle='dashed')
    ax1.legend(loc='center left', bbox_to_anchor=(1, 0.5), frameon=False)
    sns.despine(trim=True)

 # -----------------

    # Contrast 0 probability 0.2
    f, (ax1) = plt.subplots(1, 1, sharey=True, sharex=False, dpi=dpi)
    lineplt = sns.lineplot(x='time', y='baseline_subtracted', hue='Stim_side', data=pupil_size[((pupil_size['contrast'] == -0) & (pupil_size['probabilityLeft'] == 0.2)) | ((pupil_size['contrast'] == 0) & (pupil_size['probabilityLeft'] == 0.2))], legend='full', ci=68, ax=ax1, estimator=np.median, palette = sns.color_palette(colors))
    ax1.set(xlabel='Time relative to Feedback Times (s)', ylabel='Pupil size (%)', title=f' Contrast = 0 and probability 0.2' '    ' 'n=20 mice', ylim=[-15, 15])
    plt.axvline(x = 0, color = 'black', label = 'Feedback Times', linestyle='dashed')
    ax1.legend(loc='center left', bbox_to_anchor=(1, 0.5), frameon=False)
    sns.despine(trim=True)

    # Contrast 0 probability 0.5
    f, (ax1) = plt.subplots(1, 1, sharey=True, sharex=False, dpi=dpi)
    lineplt = sns.lineplot(x='time', y='baseline_subtracted', hue='Stim_side', data=pupil_size[((pupil_size['contrast'] == -0) & (pupil_size['probabilityLeft'] == 0.5)) | ((pupil_size['contrast'] == 0) & (pupil_size['probabilityLeft'] == 0.5))], legend='full', ci=68, ax=ax1, estimator=np.median, palette = sns.color_palette(colors))
    ax1.set(xlabel='Time relative to Feedback Times (s)', ylabel='Pupil size (%)', title=f' Contrast = 0 and probability 0.5' '    ' 'n=20 mice', ylim=[-15, 15])
    plt.axvline(x = 0, color = 'black', label = 'Feedback Times', linestyle='dashed')
    ax1.legend(loc='center left', bbox_to_anchor=(1, 0.5), frameon=False)
    sns.despine(trim=True)

    # Contrast 0 probability 0.8
    f, (ax1) = plt.subplots(1, 1, sharey=True, sharex=False, dpi=dpi)
    lineplt = sns.lineplot(x='time', y='baseline_subtracted', hue='Stim_side', data=pupil_size[((pupil_size['contrast'] == -0) & (pupil_size['probabilityLeft'] == 0.8)) | ((pupil_size['contrast'] == 0) & (pupil_size['probabilityLeft'] == 0.8))], legend='full', ci=68, ax=ax1, estimator=np.median, palette = sns.color_palette(colors))
    ax1.set(xlabel='Time relative to Feedback Times (s)', ylabel='Pupil size (%)', title=f' Contrast = 0 and probability 0.8' '    ' 'n=20 mice', ylim=[-15, 15])
    plt.axvline(x = 0, color = 'black', label = 'Feedback Times', linestyle='dashed')
    ax1.legend(loc='center left', bbox_to_anchor=(1, 0.5), frameon=False)
    sns.despine(trim=True)



def all_contrasts_all_blocks_correct_error_by_stim_side_figure(pupil_size_df, subject):
   
    
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


def n_trials_choice(pupil_size_df, subject):

    pupil_size = pupil_size_df

    pupil_size = pupil_size.reset_index(drop=True)
    dpi = figure_style()
    colors = ['#F96E46', '#8E4162', '#1DC7C6', '#C89AFF']
    

    #['#47BFD1', #4D8B31', '#FF9561']

    #when the stim appears in the side with smaller probability

    # Contrast = 1
    df_slice = pupil_size_df[((pupil_size_df['Stim_side'] == -1) & (pupil_size_df['probabilityLeft'] == 0.8) & (pupil_size_df['contrast'] == 1))
                      | ((pupil_size_df['Stim_side'] == 1) & (pupil_size_df['probabilityLeft'] == 0.2) & (pupil_size_df['contrast'] == 1))].reset_index()

    f, (ax1) = plt.subplots(1, 1, sharey=True, sharex=False, dpi=dpi)
    lineplt = sns.lineplot(x='time', y='baseline_subtracted', hue='after_switch', data=df_slice, legend='full',
                        ci=68, ax=ax1, estimator=np.median, palette = sns.color_palette(colors))
    ax1.set(xlabel='Time relative to Feedback Times (s)', ylabel='Pupil size (%)', title=f' Contrast = 1''    ' 'n=20 mice', ylim=[-15, 15])
    plt.axvline(x = 0, color = 'black', label = 'Feedback Times', linestyle='dashed')
    ax1.legend(loc='center left', bbox_to_anchor=(1, 0.5), frameon=False) # Put a legend to the right of the current axis
    sns.despine(trim=True)

    # Contrast = 0.0625
    df_slice = pupil_size_df[((pupil_size_df['Stim_side'] == -1) & (pupil_size_df['probabilityLeft'] == 0.8) & (pupil_size_df['contrast'] == 0.0625))
                      | ((pupil_size_df['Stim_side'] == 1) & (pupil_size_df['probabilityLeft'] == 0.2) & (pupil_size_df['contrast'] == 0.0625))].reset_index()

    f, (ax1) = plt.subplots(1, 1, sharey=True, sharex=False, dpi=dpi)
    lineplt = sns.lineplot(x='time', y='baseline_subtracted', hue='after_switch', data=df_slice, legend='full',
                        ci=68, ax=ax1, estimator=np.median, palette = sns.color_palette(colors))
    ax1.set(xlabel='Time relative to Feedback Times (s)', ylabel='Pupil size (%)', title=f' Contrast = 0.0625''    ' 'n=20 mice', ylim=[-15, 15])
    plt.axvline(x = 0, color = 'black', label = 'Feedback Times', linestyle='dashed')
    ax1.legend(loc='center left', bbox_to_anchor=(1, 0.5), frameon=False) # Put a legend to the right of the current axis
    sns.despine(trim=True)
    
    # Contrast = 0
    df_slice = pupil_size_df[((pupil_size_df['Stim_side'] == -1) & (pupil_size_df['probabilityLeft'] == 0.8) & (pupil_size_df['contrast'] == 0))
                      | ((pupil_size_df['Stim_side'] == 1) & (pupil_size_df['probabilityLeft'] == 0.2) & (pupil_size_df['contrast'] == 0))].reset_index()
    
    f, (ax1) = plt.subplots(1, 1, sharey=True, sharex=False, dpi=dpi)
    lineplt = sns.lineplot(x='time', y='baseline_subtracted', hue='after_switch', data=df_slice, legend='full',
                        ci=68, ax=ax1, estimator=np.median, palette = sns.color_palette(colors))
    ax1.set(xlabel='Time relative to Feedback Times (s)', ylabel='Pupil size (%)', title=f' Contrast = 0''    ' 'n=20 mice', ylim=[-15, 15])
    plt.axvline(x = 0, color = 'black', label = 'Feedback Times', linestyle='dashed')
    ax1.legend(loc='center left', bbox_to_anchor=(1, 0.5), frameon=False) # Put a legend to the right of the current axis
    sns.despine(trim=True)
    
    # No matter the contrast
    df_slice = pupil_size[((pupil_size['Stim_side'] == -1) & (pupil_size['probabilityLeft'] == 0.8))
                          | ((pupil_size['Stim_side'] == 1) & (pupil_size['probabilityLeft'] == 0.2))].reset_index()

    f, (ax1) = plt.subplots(1, 1, sharey=True, sharex=False, dpi=dpi)
    lineplt = sns.lineplot(x='time', y='baseline_subtracted', hue='after_switch', data=df_slice, legend='full',
                        ci=68, ax=ax1, estimator=np.median, palette = sns.color_palette(colors))
    ax1.set(xlabel='Time relative to Feedback Times (s)', ylabel='Pupil size (%)', title=f' Contrast = All''    ' 'n=20 mice', ylim=[-15, 15])
    plt.axvline(x = 0, color = 'black', label = 'Feedback Times', linestyle='dashed')
    ax1.legend(loc='center left', bbox_to_anchor=(1, 0.5), frameon=False) # Put a legend to the right of the current axis
    sns.despine(trim=True)
    

#%%

def all_contrasts_all_blocks_correct_error_by_stim_side_figure_clean(pupil_size_df, subject):
   
    
    pupil_size = pupil_size_df
    
    
    # CORRECT vs INCORRECT TRIALS
    pupil_size = pupil_size.reset_index(drop=True)

    dpi = figure_style()
    colors = ['#47BFD1', '#C89AFF']


    # Contrast 1 probability 0.2 - CORRECT TRIALS
    f, (ax1) = plt.subplots(1, 1, sharey=True, sharex=False, dpi=dpi)
    lineplt = sns.lineplot(x='time', y='baseline_subtracted', hue='Stim_side', data=pupil_size[((pupil_size['contrast'] == -1) & (pupil_size['probabilityLeft'] == 0.2) & (pupil_size['Feedback_type'] == 1)) | ((pupil_size['contrast'] == 1) & (pupil_size['probabilityLeft'] == 0.2) & (pupil_size['Feedback_type'] == 1))], legend='full', ci=68, ax=ax1, estimator=np.median, palette = sns.color_palette(colors))
    ax1.set(xlabel='Time relative to Feedback Times (s)', ylabel='Pupil size (%)', title=f' Contrast = 1 and probability 0.2 CORRECT', ylim=[-15, 15])
    ax1.plot([0, 0], ax1.get_ylim(), ls='--', color='black', label='Feedback Times')
    ax1.legend(loc='center left', bbox_to_anchor=(1, 0.5), frameon=False)
    sns.despine(trim=True)

    # Contrast 1 probability 0.2 - INCORRECT TRIALS
    f, (ax1) = plt.subplots(1, 1, sharey=True, sharex=False, dpi=dpi)
    sns.lineplot(x='time', y='baseline_subtracted', hue='Stim_side', data=pupil_size[((pupil_size['contrast'] == -1) & (pupil_size['probabilityLeft'] == 0.2) & (pupil_size['Feedback_type'] == -1)) | ((pupil_size['contrast'] == 1) & (pupil_size['probabilityLeft'] == 0.2) & (pupil_size['Feedback_type'] == -1))], legend='full', ci=68, ax=ax1, estimator=np.median, palette = sns.color_palette(colors))
    ax1.set(xlabel='Time relative to Feedback Times (s)', ylabel='Pupil size (%)', title=f' Contrast = 1 and probability 0.2 INCORRECT', ylim=[-15, 15])
    ax1.plot([0, 0], ax1.get_ylim(), ls='--', color='black', label='Feedback Times')
    ax1.legend(loc='center left', bbox_to_anchor=(1, 0.5), frameon=False)
    sns.despine(trim=True)

    # Contrast 1 probability 0.5 - CORRECT TRIALS
    f, (ax1) = plt.subplots(1, 1, sharey=True, sharex=False, dpi=dpi)
    sns.lineplot(x='time', y='baseline_subtracted', hue='Stim_side', data=pupil_size[((pupil_size['contrast'] == -1) & (pupil_size['probabilityLeft'] == 0.5) & (pupil_size['Feedback_type'] == 1)) | ((pupil_size['contrast'] == 1) & (pupil_size['probabilityLeft'] == 0.5) & (pupil_size['Feedback_type'] == 1))], legend='full', ci=68, ax=ax1, estimator=np.median, palette = sns.color_palette(colors))
    ax1.set(xlabel='Time relative to Feedback Times (s)', ylabel='', title=f' Contrast = 1 and probability 0.5 CORRECT', ylim=[-15, 15])
    ax1.plot([0, 0], ax1.get_ylim(), ls='--', color='black', label='Feedback Times')
    ax1.legend(loc='center left', bbox_to_anchor=(1, 0.5), frameon=False)
    sns.despine(trim=True)

    # Contrast 1 probability 0.5 - INCORRECT TRIALS
    f, (ax1) = plt.subplots(1, 1, sharey=True, sharex=False, dpi=dpi)
    sns.lineplot(x='time', y='baseline_subtracted', hue='Stim_side', data=pupil_size[((pupil_size['contrast'] == -1) & (pupil_size['probabilityLeft'] == 0.5) & (pupil_size['Feedback_type'] == -1)) | ((pupil_size['contrast'] == 1) & (pupil_size['probabilityLeft'] == 0.5) & (pupil_size['Feedback_type'] == -1))], legend='full', ci=68, ax=ax1, estimator=np.median, palette = sns.color_palette(colors))
    ax1.set(xlabel='Time relative to Feedback Times (s)', ylabel='', title=f' Contrast = 1 and probability 0.5 INCORRECT', ylim=[-15, 15])
    ax1.plot([0, 0], ax1.get_ylim(), ls='--', color='black', label='Feedback Times')
    ax1.legend(loc='center left', bbox_to_anchor=(1, 0.5), frameon=False)
    sns.despine(trim=True)

    # Contrast 1 probability 0.8 - CORRECT TRIALS
    f, (ax1) = plt.subplots(1, 1, sharey=True, sharex=False, dpi=dpi)
    sns.lineplot(x='time', y='baseline_subtracted', hue='Stim_side', data=pupil_size[((pupil_size['contrast'] == -1) & (pupil_size['probabilityLeft'] == 0.8) & (pupil_size['Feedback_type'] == 1)) | ((pupil_size['contrast'] == 1) & (pupil_size['probabilityLeft'] == 0.8) & (pupil_size['Feedback_type'] == 1))], legend='full', ci=68, ax=ax1, estimator=np.median, palette = sns.color_palette(colors))
    ax1.set(xlabel='Time relative to Feedback Times (s)', ylabel='', title=f' Contrast = 1 and probability 0.8 CORRECT', ylim=[-15, 15])
    ax1.plot([0, 0], ax1.get_ylim(), ls='--', color='black', label='Feedback Times')
    ax1.legend(loc='center left', bbox_to_anchor=(1, 0.5), frameon=False)
    sns.despine(trim=True)

    # Contrast 1 probability 0.8 - INCORRECT TRIALS
    f, (ax1) = plt.subplots(1, 1, sharey=True, sharex=False, dpi=dpi)
    sns.lineplot(x='time', y='baseline_subtracted', hue='Stim_side', data=pupil_size[((pupil_size['contrast'] == -1) & (pupil_size['probabilityLeft'] == 0.8) & (pupil_size['Feedback_type'] == -1)) | ((pupil_size['contrast'] == 1) & (pupil_size['probabilityLeft'] == 0.8) & (pupil_size['Feedback_type'] == -1))], legend='full', ci=68, ax=ax1, estimator=np.median, palette = sns.color_palette(colors))
    ax1.set(xlabel='Time relative to Feedback Times (s)', ylabel='', title=f' Contrast = 1 and probability 0.8 INCORRECT', ylim=[-15, 15])
    ax1.plot([0, 0], ax1.get_ylim(), ls='--', color='black', label='Feedback Times')
    ax1.legend(loc='center left', bbox_to_anchor=(1, 0.5), frameon=False)
    sns.despine(trim=True)


 # ----------


    # Contrast 0.25 probability 0.2 - CORRECT TRIALS
    f, (ax1) = plt.subplots(1, 1, sharey=True, sharex=False, dpi=dpi)
    sns.lineplot(x='time', y='baseline_subtracted', hue='Stim_side', data=pupil_size[((pupil_size['contrast'] == -0.25) & (pupil_size['probabilityLeft'] == 0.2) & (pupil_size['Feedback_type'] == 1)) | ((pupil_size['contrast'] == 0.25) & (pupil_size['probabilityLeft'] == 0.2) & (pupil_size['Feedback_type'] == 1))], legend='full', ci=68, ax=ax1, estimator=np.median, palette = sns.color_palette(colors))
    ax1.set(xlabel='Time relative to Feedback Times (s)', ylabel='Pupil size (%)', title=f' Contrast = 0.25 and probability 0.2 CORRECT', ylim=[-15, 15])
    ax1.plot([0, 0], ax1.get_ylim(), ls='--', color='black', label='Feedback Times')
    ax1.legend(loc='center left', bbox_to_anchor=(1, 0.5), frameon=False)
    sns.despine(trim=True)

    # Contrast 0.25 probability 0.2 - INCORRECT TRIALS
    f, (ax1) = plt.subplots(1, 1, sharey=True, sharex=False, dpi=dpi)
    sns.lineplot(x='time', y='baseline_subtracted', hue='Stim_side', data=pupil_size[((pupil_size['contrast'] == -0.25) & (pupil_size['probabilityLeft'] == 0.2) & (pupil_size['Feedback_type'] == -1)) | ((pupil_size['contrast'] == 0.25) & (pupil_size['probabilityLeft'] == 0.2) & (pupil_size['Feedback_type'] == -1))], legend='full', ci=68, ax=ax1, estimator=np.median, palette = sns.color_palette(colors))
    ax1.set(xlabel='Time relative to Feedback Times (s)', ylabel='Pupil size (%)', title=f' Contrast = 0.25 and probability 0.2 INCORRECT', ylim=[-15, 15])
    ax1.plot([0, 0], ax1.get_ylim(), ls='--', color='black', label='Feedback Times')
    ax1.legend(loc='center left', bbox_to_anchor=(1, 0.5), frameon=False)
    sns.despine(trim=True)

    # Contrast 0.25 probability 0.5 - CORRECT TRIALS
    f, (ax1) = plt.subplots(1, 1, sharey=True, sharex=False, dpi=dpi)
    sns.lineplot(x='time', y='baseline_subtracted', hue='Stim_side', data=pupil_size[((pupil_size['contrast'] == -0.25) & (pupil_size['probabilityLeft'] == 0.5) & (pupil_size['Feedback_type'] == 1)) | ((pupil_size['contrast'] == 0.25) & (pupil_size['probabilityLeft'] == 0.5) & (pupil_size['Feedback_type'] == 1))], legend='full', ci=68, ax=ax1, estimator=np.median, palette = sns.color_palette(colors))
    ax1.set(xlabel='Time relative to Feedback Times (s)', ylabel='', title=f' Contrast = 0.25 and probability 0.5 CORRECT', ylim=[-15, 15])
    ax1.plot([0, 0], ax1.get_ylim(), ls='--', color='black', label='Feedback Times')
    ax1.legend(loc='center left', bbox_to_anchor=(1, 0.5), frameon=False)
    sns.despine(trim=True)

    # Contrast 0.25 probability 0.5 - INCORRECT TRIALS
    f, (ax1) = plt.subplots(1, 1, sharey=True, sharex=False, dpi=dpi)
    sns.lineplot(x='time', y='baseline_subtracted', hue='Stim_side', data=pupil_size[((pupil_size['contrast'] == -0.25) & (pupil_size['probabilityLeft'] == 0.5) & (pupil_size['Feedback_type'] == -1)) | ((pupil_size['contrast'] == 0.25) & (pupil_size['probabilityLeft'] == 0.5) & (pupil_size['Feedback_type'] == -1))], legend='full', ci=68, ax=ax1, estimator=np.median, palette = sns.color_palette(colors))
    ax1.set(xlabel='Time relative to Feedback Times (s)', ylabel='', title=f' Contrast = 0.25 and probability 0.5 INCORRECT', ylim=[-15, 15])
    ax1.plot([0, 0], ax1.get_ylim(), ls='--', color='black', label='Stim Onset')
    ax1.legend(loc='center left', bbox_to_anchor=(1, 0.5), frameon=False)
    sns.despine(trim=True)

    # Contrast 0.25 probability 0.8 - CORRECT TRIALS
    f, (ax1) = plt.subplots(1, 1, sharey=True, sharex=False, dpi=dpi)
    sns.lineplot(x='time', y='baseline_subtracted', hue='Stim_side', data=pupil_size[((pupil_size['contrast'] == -0.25) & (pupil_size['probabilityLeft'] == 0.8) & (pupil_size['Feedback_type'] == 1)) | ((pupil_size['contrast'] == 0.25) & (pupil_size['probabilityLeft'] == 0.8) & (pupil_size['Feedback_type'] == 1))], legend='full', ci=68, ax=ax1, estimator=np.median, palette = sns.color_palette(colors))
    ax1.set(xlabel='Time relative to Feedback Times (s)', ylabel='', title=f' Contrast = 0.25 and probability 0.8 CORRECT', ylim=[-15, 15])
    ax1.plot([0, 0], ax1.get_ylim(), ls='--', color='black', label='Feedback Times')
    ax1.legend(loc='center left', bbox_to_anchor=(1, 0.5), frameon=False)
    sns.despine(trim=True)

    # Contrast 0.25 probability 0.8 - INCORRECT TRIALS
    f, (ax1) = plt.subplots(1, 1, sharey=True, sharex=False, dpi=dpi)
    sns.lineplot(x='time', y='baseline_subtracted', hue='Stim_side', data=pupil_size[((pupil_size['contrast'] == -0.25) & (pupil_size['probabilityLeft'] == 0.8) & (pupil_size['Feedback_type'] == -1)) | ((pupil_size['contrast'] == 0.25) & (pupil_size['probabilityLeft'] == 0.8) & (pupil_size['Feedback_type'] == -1))], legend='full', ci=68, ax=ax1, estimator=np.median, palette = sns.color_palette(colors))
    ax1.set(xlabel='Time relative to Feedback Times (s)', ylabel='', title=f' Contrast = 0.25 and probStim Onsetability 0.8 INCORRECT', ylim=[-15, 15])
    ax1.plot([0, 0], ax1.get_ylim(), ls='--', color='black', label='Feedback Times')
    ax1.legend(loc='center left', bbox_to_anchor=(1, 0.5), frameon=False)
    sns.despine(trim=True)


# ------------


    # Contrast 0.125 probability 0.2 - CORRECT TRIALS
    f, (ax1) = plt.subplots(1, 1, sharey=True, sharex=False, dpi=dpi)
    sns.lineplot(x='time', y='baseline_subtracted', hue='Stim_side', data=pupil_size[((pupil_size['contrast'] == -0.125) & (pupil_size['probabilityLeft'] == 0.2) & (pupil_size['Feedback_type'] == 1)) | ((pupil_size['contrast'] == 0.125) & (pupil_size['probabilityLeft'] == 0.2) & (pupil_size['Feedback_type'] == 1))], legend='full', ci=68, ax=ax1, estimator=np.median, palette = sns.color_palette(colors))
    ax1.set(xlabel='Time relative to Feedback Times (s)', ylabel='Pupil size (%)', title=f' Contrast = 0.125 and probability 0.2 CORRECT', ylim=[-15, 15])
    ax1.plot([0, 0], ax1.get_ylim(), ls='--', color='black', label='Feedback Times')
    ax1.legend(loc='center left', bbox_to_anchor=(1, 0.5), frameon=False)
    sns.despine(trim=True)

    # Contrast 0.125 probability 0.2 - INCORRECT TRIALS
    f, (ax1) = plt.subplots(1, 1, sharey=True, sharex=False, dpi=dpi)
    sns.lineplot(x='time', y='baseline_subtracted', hue='Stim_side', data=pupil_size[((pupil_size['contrast'] == -0.125) & (pupil_size['probabilityLeft'] == 0.2) & (pupil_size['Feedback_type'] == -1)) | ((pupil_size['contrast'] == 0.125) & (pupil_size['probabilityLeft'] == 0.2) & (pupil_size['Feedback_type'] == -1))], legend='full', ci=68, ax=ax1, estimator=np.median, palette = sns.color_palette(colors))
    ax1.set(xlabel='Time relative to Feedback Times (s)', ylabel='Pupil size (%)', title=f' ContrastStim Onset = 0.125 and probability 0.2 INCORRECT', ylim=[-15, 15])
    ax1.plot([0, 0], ax1.get_ylim(), ls='--', color='black', label='Feedback Times')
    ax1.legend(loc='center left', bbox_to_anchor=(1, 0.5), frameon=False)
    sns.despine(trim=True)

    # Contrast 0.125 probability 0.5 - CORRECT TRIALS
    f, (ax1) = plt.subplots(1, 1, sharey=True, sharex=False, dpi=dpi)
    sns.lineplot(x='time', y='baseline_subtracted', hue='Stim_side', data=pupil_size[((pupil_size['contrast'] == -0.125) & (pupil_size['probabilityLeft'] == 0.5) & (pupil_size['Feedback_type'] == 1)) | ((pupil_size['contrast'] == 0.125) & (pupil_size['probabilityLeft'] == 0.5) & (pupil_size['Feedback_type'] == 1))], legend='full', ci=68, ax=ax1, estimator=np.median, palette = sns.color_palette(colors))
    ax1.set(xlabel='Time relative to Feedback Times (s)', ylabel='', title=f' Contrast = 0.125 and probability 0.5 CORRECT', ylim=[-15, 15])
    ax1.plot([0, 0], ax1.get_ylim(), ls='--', color='black', label='Feedback Times')
    ax1.legend(loc='center left', bbox_to_anchor=(1, 0.5), frameon=False)
    sns.despine(trim=True)

    # Contrast 0.125 probability 0.5 - INCORRECT TRIALS
    f, (ax1) = plt.subplots(1, 1, sharey=True, sharex=False, dpi=dpi)
    sns.lineplot(x='time', y='baseline_subtracted', hue='Stim_side', data=pupil_size[((pupil_size['contrast'] == -0.125) & (pupil_size['probabilityLeft'] == 0.5) & (pupil_size['Feedback_type'] == -1)) | ((pupil_size['contrast'] == 0.125) & (pupil_size['probabilityLeft'] == 0.5) & (pupil_size['Feedback_type'] == -1))], legend='full', ci=68, ax=ax1, estimator=np.median, palette = sns.color_palette(colors))
    ax1.set(xlabel='Time relative to Feedback Times (s)', ylabel='', title=f' Contrast = 0.125 and probability 0.5 INCORRECT', ylim=[-15, 15])
    ax1.plot([0, 0], ax1.get_ylim(), ls='--', color='black', label='Feedback Times')
    ax1.legend(loc='center left', bbox_to_anchor=(1, 0.5), frameon=False)
    sns.despine(trim=True)

    # Contrast 0.125 probability 0.8 - CORRECT TRIALS
    f, (ax1) = plt.subplots(1, 1, sharey=True, sharex=False, dpi=dpi)
    sns.lineplot(x='time', y='baseline_subtracted', hue='Stim_side', data=pupil_size[((pupil_size['contrast'] == -0.125) & (pupil_size['probabilityLeft'] == 0.8) & (pupil_size['Feedback_type'] == 1)) | ((pupil_size['contrast'] == 0.125) & (pupil_size['probabilityLeft'] == 0.8) & (pupil_size['Feedback_type'] == 1))], legend='full', ci=68, ax=ax1, estimator=np.median, palette = sns.color_palette(colors))
    ax1.set(xlabel='Time relative to Feedback Times (s)', ylabel='', title=f' Contrast = 0.125 and probability 0.8 CORRECT', ylim=[-15, 15])
    ax1.plot([0, 0], ax1.get_ylim(), ls='--', color='black', label='Feedback Times')
    ax1.legend(loc='center left', bbox_to_anchor=(1, 0.5), frameon=False)
    sns.despine(trim=True)

    # Contrast 0.125 probability 0.8 - INCORRECT TRIALS
    f, (ax1) = plt.subplots(1, 1, sharey=True, sharex=False, dpi=dpi)
    sns.lineplot(x='time', y='baseline_subtracted', hue='Stim_side', data=pupil_size[((pupil_size['contrast'] == -0.125) & (pupil_size['probabilityLeft'] == 0.8) & (pupil_size['Feedback_type'] == -1)) | ((pupil_size['contrast'] == 0.125) & (pupil_size['probabilityLeft'] == 0.8) & (pupil_size['Feedback_type'] == -1))], legend='full', ci=68, ax=ax1, estimator=np.median, palette = sns.color_palette(colors))
    ax1.set(xlabel='Time relative to Feedback Times (s)', ylabel='', title=f' Contrast = 0.125 and probability 0.8 INCORRECT', ylim=[-15, 15])
    ax1.plot([0, 0], ax1.get_ylim(), ls='--', color='black', label='Feedback Times')
    ax1.legend(loc='center left', bbox_to_anchor=(1, 0.5), frameon=False)
    sns.despine(trim=True)

# ------------


    # Contrast 0.0625 probability 0.2 - CORRECT TRIALS
    f, (ax1) = plt.subplots(1, 1, sharey=True, sharex=False, dpi=dpi)
    sns.lineplot(x='time', y='baseline_subtracted', hue='Stim_side', data=pupil_size[((pupil_size['contrast'] == -0.0625) & (pupil_size['probabilityLeft'] == 0.2) & (pupil_size['Feedback_type'] == 1)) | ((pupil_size['contrast'] == 0.0625) & (pupil_size['probabilityLeft'] == 0.2) & (pupil_size['Feedback_type'] == 1))], legend='full', ci=68, ax=ax1, estimator=np.median, palette = sns.color_palette(colors))
    ax1.set(xlabel='Time relative to Feedback Times (s)', ylabel='Pupil size (%)', title=f' Contrast = 0.0625 and probability 0.2 CORRECT', ylim=[-15, 15])
    ax1.plot([0, 0], ax1.get_ylim(), ls='--', color='black', label='Feedback Times')
    ax1.legend(loc='center left', bbox_to_anchor=(1, 0.5), frameon=False)
    sns.despine(trim=True)

    # Contrast 0.0625 probability 0.2 - INCORRECT TRIALS
    f, (ax1) = plt.subplots(1, 1, sharey=True, sharex=False, dpi=dpi)
    sns.lineplot(x='time', y='baseline_subtracted', hue='Stim_side', data=pupil_size[((pupil_size['contrast'] == -0.0625) & (pupil_size['probabilityLeft'] == 0.2) & (pupil_size['Feedback_type'] == -1)) | ((pupil_size['contrast'] == 0.0625) & (pupil_size['probabilityLeft'] == 0.2) & (pupil_size['Feedback_type'] == -1))], legend='full', ci=68, ax=ax1, estimator=np.median, palette = sns.color_palette(colors))
    ax1.set(xlabel='Time relative to Feedback Times (s)', ylabel='Pupil size (%)', title=f' Contrast = 0.0625 and probability 0.2 INCORRECT', ylim=[-15, 15])
    ax1.plot([0, 0], ax1.get_ylim(), ls='--', color='black', label='Feedback Times')
    ax1.legend(loc='center left', bbox_to_anchor=(1, 0.5), frameon=False)
    sns.despine(trim=True)

    # Contrast 0.0625 probability 0.5 - CORRECT TRIALS
    f, (ax1) = plt.subplots(1, 1, sharey=True, sharex=False, dpi=dpi)
    sns.lineplot(x='time', y='baseline_subtracted', hue='Stim_side', data=pupil_size[((pupil_size['contrast'] == -0.0625) & (pupil_size['probabilityLeft'] == 0.5) & (pupil_size['Feedback_type'] == 1)) | ((pupil_size['contrast'] == 0.0625) & (pupil_size['probabilityLeft'] == 0.5) & (pupil_size['Feedback_type'] == 1))], legend='full', ci=68, ax=ax1, estimator=np.median, palette = sns.color_palette(colors))
    ax1.set(xlabel='Time relative to Feedback Times (s)', ylabel='', title=f' Contrast = 0.0625 and probability 0.5 CORRECT', ylim=[-15, 15])
    ax1.plot([0, 0], ax1.get_ylim(), ls='--', color='black', label='Feedback Times')
    ax1.legend(loc='center left', bbox_to_anchor=(1, 0.5), frameon=False)
    sns.despine(trim=True)

    # Contrast 0.0625 probability 0.5 - INCORRECT TRIALS
    f, (ax1) = plt.subplots(1, 1, sharey=True, sharex=False, dpi=dpi)
    sns.lineplot(x='time', y='baseline_subtracted', hue='Stim_side', data=pupil_size[((pupil_size['contrast'] == -0.0625) & (pupil_size['probabilityLeft'] == 0.5) & (pupil_size['Feedback_type'] == -1)) | ((pupil_size['contrast'] == 0.0625) & (pupil_size['probabilityLeft'] == 0.5) & (pupil_size['Feedback_type'] == -1))], legend='full', ci=68, ax=ax1, estimator=np.median, palette = sns.color_palette(colors))
    ax1.set(xlabel='Time relative to Feedback Times (s)', ylabel='', title=f' Contrast = 0.0625 and probability 0.5 INCORRECT', ylim=[-15, 15])
    ax1.plot([0, 0], ax1.get_ylim(), ls='--', color='black', label='Feedback Times')
    ax1.legend(loc='center left', bbox_to_anchor=(1, 0.5), frameon=False)
    sns.despine(trim=True)

    # Contrast 0.0625 probability 0.8 - CORRECT TRIALS
    f, (ax1) = plt.subplots(1, 1, sharey=True, sharex=False, dpi=dpi)
    sns.lineplot(x='time', y='baseline_subtracted', hue='Stim_side', data=pupil_size[((pupil_size['contrast'] == -0.0625) & (pupil_size['probabilityLeft'] == 0.8) & (pupil_size['Feedback_type'] == 1)) | ((pupil_size['contrast'] == 0.0625) & (pupil_size['probabilityLeft'] == 0.8) & (pupil_size['Feedback_type'] == 1))], legend='full', ci=68, ax=ax1, estimator=np.median, palette = sns.color_palette(colors))
    ax1.set(xlabel='Time relative to Feedback Times (s)', ylabel='', title=f' Contrast = 0.0625 and probability 0.8 CORRECT', ylim=[-15, 15])
    ax1.plot([0, 0], ax1.get_ylim(), ls='--', color='black', label='Feedback Times')
    ax1.legend(loc='center left', bbox_to_anchor=(1, 0.5), frameon=False)
    sns.despine(trim=True)

    # Contrast 0.0625 probability 0.8 - INCORRECT TRIALS
    f, (ax1) = plt.subplots(1, 1, sharey=True, sharex=False, dpi=dpi)
    sns.lineplot(x='time', y='baseline_subtracted', hue='Stim_side', data=pupil_size[((pupil_size['contrast'] == -0.0625) & (pupil_size['probabilityLeft'] == 0.8) & (pupil_size['Feedback_type'] == -1)) | ((pupil_size['contrast'] == 0.0625) & (pupil_size['probabilityLeft'] == 0.8) & (pupil_size['Feedback_type'] == -1))], legend='full', ci=68, ax=ax1, estimator=np.median, palette = sns.color_palette(colors))
    ax1.set(xlabel='Time relative to Feedback Times (s)', ylabel='', title=f' Contrast = 0.0625 and probability 0.8 INCORRECT', ylim=[-15, 15])
    ax1.plot([0, 0], ax1.get_ylim(), ls='--', color='black', label='Feedback Times')
    ax1.legend(loc='center left', bbox_to_anchor=(1, 0.5), frameon=False)
    sns.despine(trim=True)

# ------------


    # Contrast 0 probability 0.2 - CORRECT TRIALS
    f, (ax1) = plt.subplots(1, 1, sharey=True, sharex=False, dpi=dpi)
    sns.lineplot(x='time', y='baseline_subtracted', hue='Stim_side', data=pupil_size[((pupil_size['contrast'] == -0) & (pupil_size['probabilityLeft'] == 0.2) & (pupil_size['Feedback_type'] == 1)) | ((pupil_size['contrast'] == 0) & (pupil_size['probabilityLeft'] == 0.2) & (pupil_size['Feedback_type'] == 1))], legend='full', ci=68, ax=ax1, estimator=np.median, palette = sns.color_palette(colors))
    ax1.set(xlabel='Time relative to Feedback Times (s)', ylabel='Pupil size (%)', title=f' Contrast = 0 and probability 0.2 CORRECT', ylim=[-15, 15])
    ax1.plot([0, 0], ax1.get_ylim(), ls='--', color='black', label='Feedback Times')
    ax1.legend(loc='center left', bbox_to_anchor=(1, 0.5), frameon=False)
    sns.despine(trim=True)

    # Contrast 0 probability 0.2 - INCORRECT TRIALS
    f, (ax1) = plt.subplots(1, 1, sharey=True, sharex=False, dpi=dpi)
    sns.lineplot(x='time', y='baseline_subtracted', hue='Stim_side', data=pupil_size[((pupil_size['contrast'] == -0) & (pupil_size['probabilityLeft'] == 0.2) & (pupil_size['Feedback_type'] == -1)) | ((pupil_size['contrast'] == 0) & (pupil_size['probabilityLeft'] == 0.2) & (pupil_size['Feedback_type'] == -1))], legend='full', ci=68, ax=ax1, estimator=np.median, palette = sns.color_palette(colors))
    ax1.set(xlabel='Time relative to Feedback Times (s)', ylabel='Pupil size (%)', title=f' Contrast = 0 and probability 0.2 INCORRECT', ylim=[-15, 15])
    ax1.plot([0, 0], ax1.get_ylim(), ls='--', color='black', label='Feedback Times')
    ax1.legend(loc='center left', bbox_to_anchor=(1, 0.5), frameon=False)
    sns.despine(trim=True)

    # Contrast 0 probability 0.5 - CORRECT TRIALS
    f, (ax1) = plt.subplots(1, 1, sharey=True, sharex=False, dpi=dpi)
    sns.lineplot(x='time', y='baseline_subtracted', hue='Stim_side', data=pupil_size[((pupil_size['contrast'] == -0) & (pupil_size['probabilityLeft'] == 0.5) & (pupil_size['Feedback_type'] == 1)) | ((pupil_size['contrast'] == 0) & (pupil_size['probabilityLeft'] == 0.5) & (pupil_size['Feedback_type'] == 1))], legend='full', ci=68, ax=ax1, estimator=np.median, palette = sns.color_palette(colors))
    ax1.set(xlabel='Time relative to Feedback Times (s)', ylabel='', title=f' Contrast = 0 and probability 0.5 CORRECT', ylim=[-15, 15])
    ax1.plot([0, 0], ax1.get_ylim(), ls='--', color='black', label='Feedback Times')
    ax1.legend(loc='center left', bbox_to_anchor=(1, 0.5), frameon=False)
    sns.despine(trim=True)

    # Contrast 0 probability 0.5 - INCORRECT TRIALS
    f, (ax1) = plt.subplots(1, 1, sharey=True, sharex=False, dpi=dpi)
    sns.lineplot(x='time', y='baseline_subtracted', hue='Stim_side', data=pupil_size[((pupil_size['contrast'] == -0) & (pupil_size['probabilityLeft'] == 0.5) & (pupil_size['Feedback_type'] == -1)) | ((pupil_size['contrast'] == 0) & (pupil_size['probabilityLeft'] == 0.5) & (pupil_size['Feedback_type'] == -1))], legend='full', ci=68, ax=ax1, estimator=np.median, palette = sns.color_palette(colors))
    ax1.set(xlabel='Time relative to Feedback Times (s)', ylabel='', title=f' Contrast = 0 and probability 0.5 INCORRECT', ylim=[-15, 15])
    ax1.plot([0, 0], ax1.get_ylim(), ls='--', color='black', label='Feedback Times')
    ax1.legend(loc='center left', bbox_to_anchor=(1, 0.5), frameon=False)
    sns.despine(trim=True)

    # Contrast 0 probability 0.8 - CORRECT TRIALS
    f, (ax1) = plt.subplots(1, 1, sharey=True, sharex=False, dpi=dpi)
    sns.lineplot(x='time', y='baseline_subtracted', hue='Stim_side', data=pupil_size[((pupil_size['contrast'] == -0) & (pupil_size['probabilityLeft'] == 0.8) & (pupil_size['Feedback_type'] == 1)) | ((pupil_size['contrast'] == 0) & (pupil_size['probabilityLeft'] == 0.8) & (pupil_size['Feedback_type'] == 1))], legend='full', ci=68, ax=ax1, estimator=np.median, palette = sns.color_palette(colors))
    ax1.set(xlabel='Time relative to Feedback Times (s)', ylabel='', title=f' Contrast = 0 and probability 0.8 CORRECT', ylim=[-15, 15])
    ax1.plot([0, 0], ax1.get_ylim(), ls='--', color='black', label='Feedback Times')
    ax1.legend(loc='center left', bbox_to_anchor=(1, 0.5), frameon=False)
    sns.despine(trim=True)
    
    # Contrast 0 probability 0.8 - INCORRECT TRIALS
    f, (ax1) = plt.subplots(1, 1, sharey=True, sharex=False, dpi=dpi)
    sns.lineplot(x='time', y='baseline_subtracted', hue='Stim_side', data=pupil_size[((pupil_size['contrast'] == -0) & (pupil_size['probabilityLeft'] == 0.8) & (pupil_size['Feedback_type'] == -1)) | ((pupil_size['contrast'] == 0) & (pupil_size['probabilityLeft'] == 0.8) & (pupil_size['Feedback_type'] == -1))], legend='full', ci=68, ax=ax1, estimator=np.median, palette = sns.color_palette(colors))
    ax1.set(xlabel='Time relative to Feedback Times (s)', ylabel='', title=f' Contrast = 0 and probability 0.8 INCORRECT', ylim=[-15, 15])
    ax1.plot([0, 0], ax1.get_ylim(), ls='--', color='black', label='Feedback Times')
    ax1.legend(loc='center left', bbox_to_anchor=(1, 0.5), frameon=False)
    sns.despine(trim=True)


 

if __name__ == '__main__':
    subject = "ZFM-02368"
    pupil_size_df = load_pupil_size_df(subject)
    all_contrasts_by_blocks(pupil_size_df)
    
    
    # all_contrasts_per_block_by_stim_side(pupil_size_df)
    # all_contrasts_all_blocks_correct_error_by_stim_side_figure(pupil_size_df)
    all_contrasts_per_block_by_stim_side(pupil_size_df)
    multi_fig_all_contrasts_by_blocks(pupil_size_df)
    all_contrasts_all_blocks_correct_error_by_stim_side_figure(pupil_size_df)

