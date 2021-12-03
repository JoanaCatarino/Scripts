# -*- coding: utf-8 -*-
"""
Created on Thu Nov 11 11:08:10 2021

@author: joana
"""

#%% Instructions to plot things

import matplotlib.pyplot as plt
import seaborn as sns

pip install matplotlib #se aparecer erro ao correr a linha acima

sns.set_theme(style="darkgrid") #podes usar outros estilos como “whitegrid” “white” “ticks”
# exemplo: Plot the responses of the 5-HT DRN neurons according to x event

# tens uma tabela/dataframe com um certo nome (“tabelaABC”, por ex) e escolher o x e y a plottar (“tempo” e “spikes”, por ex)
ax = sns.lineplot(x="tempo", y="spikes", data=tabelaABC,color='mediumseagreen', linewidth = 2, label = "spikezinhos") #escolhe a cor/linewidth/alpha (transparencia) que preferires
sns.lineplot(x="tempo", y="spikes",data=tabelaABC, color= 'darkgreen', linewidth = 0.25, alpha = 0.2, units="trial", estimator=None)  #ao correres tb esta linha, terás um plot com a junção do que quiseste plottar acima e o que estás a plottar aqui - que é o mesmo mas tens o units e estimator - o que eles fazem é plottar diferentes linhas de acordo com o que quiseres, “trial” no meu caso - ou seja, tenho uma linha de spikes por tempo por cada trial, enquanto que no caso acima ele agrupa todos os trials numa linha
sns.set(rc={'figure.figsize':(11.7,8.27)})

plt.axvline(x=0, color = "coral", alpha=0.75, linewidth = 2.5, label = "visual stim On") #se quiseres plottar uma linha vertical no x=0 <- útil para quando alinhares os dados a um evento (que começa em t=0)
plt.xlim(-25, 55) #para definires o limite do eixo xx
plt.title("TrainingTask_S5_T1 - 5-HT DRN signal", fontsize=15) 
plt.legend(loc="upper right", fontsize=12)
plt.xlabel("Frames (FR = 30fps)", fontsize=12)
plt.ylabel("Signal", fontsize=12) 
plt.xticks(fontsize=12)
plt.yticks(fontsize=12) 
plt.savefig('/Users/algures/Fig1.png')
plt.show()