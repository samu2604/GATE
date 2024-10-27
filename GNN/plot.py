# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# %%
# Data
all_omics_pna_aupr = [0.6613783225135244, 0.7247433651518764, 0.7836401790063063, 0.715873432449572, 0.6964264768626824]
all_omics_pna_auroc = [0.8241415192507805, 0.851890391952827, 0.8893827160493828, 0.8669135802469136, 0.8585185185185185]

all_omics_pna_no_omics_aupr_explainer = [0.7232010208612568, 0.6358607896247848, 0.8224289251540212, 0.7067331584203903, 0.747242323627645]
all_omics_pna_no_omics_auroc_explainer = [0.8565152040698347, 0.8054110301768991, 0.9140740740740741, 0.8374074074074075, 0.8688888888888888]


data = pd.DataFrame({
    'Run': ['PNA', 'PNA + explainer'],
    'AUROC': [np.mean(all_omics_pna_auroc), np.mean(all_omics_pna_no_omics_auroc_explainer)],
    'AUPR': [np.mean(all_omics_pna_aupr), np.mean(all_omics_pna_no_omics_aupr_explainer)]
})

data_long = data.melt(id_vars=['Run'], value_vars=['AUROC', 'AUPR'], var_name='Metric', value_name='Value')

# Set the color mapping for each metric
color_map = {'AUROC': 'darkblue', 'AUPR': 'lightblue'}

n_metrics = data_long['Metric'].nunique()
bar_width = 0.2
index = np.arange(len(data['Run']))

# Error data (example values, replace with your actual error data)
error_AUROC = [np.std(all_omics_pna_auroc), np.std(all_omics_pna_no_omics_auroc_explainer)]
error_AUPR = [np.std(all_omics_pna_aupr), np.std(all_omics_pna_no_omics_aupr_explainer)]

# Separate the data into two plots
metrics1 = ['AUROC', 'AUPR']
errors = [error_AUROC, error_AUPR] #, error_Covid, error_Antiviral]

bar_width = 0.2
# Plot 1: AUROC and AUPR
fig, ax1 = plt.subplots(figsize=(12, 12))
for i, metric in enumerate(metrics1):
    subset = data_long[data_long['Metric'] == metric]
    ax1.bar(index + i * bar_width, subset['Value'], bar_width, yerr=errors[i], label=metric, color=color_map[metric], capsize=5)

ax1.set_ylabel('Metric Values')
ax1.set_title('AUROC and AUPR for Disease Genes for PNA with explainer')
ax1.set_xticks(index + bar_width / 2)
ax1.set_xticklabels(data['Run'])
ax1.legend()

# Show plots
plt.show()

# %%

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Data
all_omics_gcn_aupr_only_viral_diseases = [0.8451082107383542, 0.9026161181722968, 0.8522040948902672, 0.7933571017302669, 0.7292424652541672]
all_omics_gcn_auroc_only_viral_diseases = [0.9314371603653601, 0.9575673488264539, 0.9224691358024693, 0.9048148148148147, 0.8462962962962962]

all_omics_gcn_aupr_all_diseases = [0.8477508422929455, 0.7909961896820995, 0.8516720349025254, 0.8652605844200871, 0.8953474545793365]
all_omics_gcn_auroc_all_diseases = [0.9437037037037038, 0.9004938271604939, 0.9231124985547462, 0.9255405249161752, 0.9606172839506172]

all_omics_mlp_aupr_all_diseases = [0.8165218736748932, 0.6889158881868592, 0.802313290037442, 0.8732081253637141, 0.547163112620471]
all_omics_mlp_auroc_all_diseases = [0.9168690021967857, 0.8515435310440513, 0.8677777777777779, 0.9653086419753087, 0.7428395061728396]

all_omics_gcn_explainer_aupr_all_diseases = [0.9357514440324443, 0.841766846672538, 0.8747136383216919, 0.8212830539168992, 0.7447915568346425]
all_omics_gcn_explainer_auroc_all_diseases = [0.966007630939993, 0.9458896982310093, 0.9450617283950616, 0.9196296296296297, 0.8741975308641976]

all_omics_pna_aupr_all_diseases = [0.9618312523878033, 0.905049014125415, 0.8941756996223588, 0.8668562568960528, 0.7992098572363879]
all_omics_pna_auroc_all_diseases = [0.976760319112036, 0.96415770609319, 0.948395061728395, 0.9317283950617284, 0.885432098765432]

# Create DataFrame
data = pd.DataFrame({
    'Run': ['GCN (Viral Diseases)', 'GCN (All Diseases)', 'MLP (All Diseases)', 'GCN + Explainer (All Diseases)', 'PNA (All Diseases)'],
    'AUROC': [
        np.mean(all_omics_gcn_auroc_only_viral_diseases),
        np.mean(all_omics_gcn_auroc_all_diseases),
        np.mean(all_omics_mlp_auroc_all_diseases),
        np.mean(all_omics_gcn_explainer_auroc_all_diseases),
        np.mean(all_omics_pna_auroc_all_diseases)
    ],
    'AUPR': [
        np.mean(all_omics_gcn_aupr_only_viral_diseases),
        np.mean(all_omics_gcn_aupr_all_diseases),
        np.mean(all_omics_mlp_aupr_all_diseases),
        np.mean(all_omics_gcn_explainer_aupr_all_diseases),
        np.mean(all_omics_pna_aupr_all_diseases)
    ]
})

data_long = data.melt(id_vars=['Run'], value_vars=['AUROC', 'AUPR'], var_name='Metric', value_name='Value')

# Set the color mapping for each metric
color_map = {'AUROC': 'darkblue', 'AUPR': 'lightblue'}

n_metrics = data_long['Metric'].nunique()
bar_width = 0.2
index = np.arange(len(data['Run']))

# Error data
error_AUROC = [
    np.std(all_omics_gcn_auroc_only_viral_diseases),
    np.std(all_omics_gcn_auroc_all_diseases),
    np.std(all_omics_mlp_auroc_all_diseases),
    np.std(all_omics_gcn_explainer_auroc_all_diseases),
    np.std(all_omics_pna_auroc_all_diseases)
]

error_AUPR = [
    np.std(all_omics_gcn_aupr_only_viral_diseases),
    np.std(all_omics_gcn_aupr_all_diseases),
    np.std(all_omics_mlp_aupr_all_diseases),
    np.std(all_omics_gcn_explainer_aupr_all_diseases),
    np.std(all_omics_pna_aupr_all_diseases)
]

# Separate the data into two plots
metrics1 = ['AUROC', 'AUPR']
errors = [error_AUROC, error_AUPR]

# Plot 1: AUROC and AUPR
fig, ax1 = plt.subplots(figsize=(12, 12))
for i, metric in enumerate(metrics1):
    subset = data_long[data_long['Metric'] == metric]
    ax1.bar(index + i * bar_width, subset['Value'], bar_width, yerr=errors[i], label=metric, color=color_map[metric], capsize=5)

ax1.set_ylabel('Metric Values')
ax1.set_title('AUROC and AUPR for Disease Genes across Different Models and Datasets')
ax1.set_xticks(index + bar_width / 2)
ax1.set_xticklabels(data['Run'])
ax1.legend()

# Show plots
plt.show()


# %%

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Data for model with reinforcement learning
model_rl_means = [0.6258314251899719, 0.4475802779197693, 0.6716932058334351, 0.5941632986068726, 0.5818707346916199]
model_rl_means_all_genes = [0.398881733417511, 0.3970126807689667, 0.45923158526420593, 0.4687679409980774, 0.47798579931259155]

# Data for model without reinforcement learning
model_no_rl_means = [0.3718365430831909, 0.5499289035797119, 0.48502135276794434, 0.5389226675033569, 0.4264296293258667]
model_no_rl_means_all_genes = [0.38675132393836975, 0.47693347930908203, 0.4516129493713379, 0.459964781999588, 0.4109734892845154]

# Calculate differences
diff_rl = [model_rl_means[i] - model_rl_means_all_genes[i] for i in range(len(model_rl_means))]
diff_no_rl = [model_no_rl_means[i] - model_no_rl_means_all_genes[i] for i in range(len(model_no_rl_means))]

# Calculate average differences
avg_diff_rl = np.mean(diff_rl)
avg_diff_no_rl = np.mean(diff_no_rl)

# Create DataFrame for plotting
data = pd.DataFrame({
    'Model': ['With RL', 'Without RL'],
    'Average Difference': [avg_diff_rl, avg_diff_no_rl]
})

# Plotting
fig, ax = plt.subplots(figsize=(8, 6))
ax.bar(data['Model'], data['Average Difference'], color=['blue', 'orange'])
ax.set_ylabel('Average Difference')
ax.set_title('Average Difference between Validated Antiviral Drug Targets and All Genes')

# Show plot
plt.show()


# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Data for model with reinforcement learning
model_rl_means = [0.6258314251899719, 0.4475802779197693, 0.6716932058334351, 0.5941632986068726, 0.5818707346916199]
model_rl_means_all_genes = [0.398881733417511, 0.3970126807689667, 0.45923158526420593, 0.4687679409980774, 0.47798579931259155]

# Data for model without reinforcement learning
model_no_rl_means = [0.3718365430831909, 0.5499289035797119, 0.48502135276794434, 0.5389226675033569, 0.4264296293258667]
model_no_rl_means_all_genes = [0.38675132393836975, 0.47693347930908203, 0.4516129493713379, 0.459964781999588, 0.4109734892845154]

# Data for PNA without reinforcement learning
model_pna_no_rl_means = [0.48365214467048645, 0.38956648111343384, 0.4774293303489685, 0.5069018602371216, 0.6312336325645447]
model_pna_no_rl_means_all_genes = [0.38285237550735474, 0.3596950173377991, 0.3943876326084137, 0.41131946444511414, 0.47379395365715027]

# Calculate differences
diff_rl = [model_rl_means[i] - model_rl_means_all_genes[i] for i in range(len(model_rl_means))]
diff_no_rl = [model_no_rl_means[i] - model_no_rl_means_all_genes[i] for i in range(len(model_no_rl_means))]
diff_pna_no_rl = [model_pna_no_rl_means[i] - model_pna_no_rl_means_all_genes[i] for i in range(len(model_pna_no_rl_means))]

# Calculate average differences
avg_diff_rl = np.mean(diff_rl)
avg_diff_no_rl = np.mean(diff_no_rl)
avg_diff_pna_no_rl = np.mean(diff_pna_no_rl)

# Create DataFrame for plotting
data = pd.DataFrame({
    'Model': ['With RL', 'Without RL', 'PNA Without RL'],
    'Average Difference': [avg_diff_rl, avg_diff_no_rl, avg_diff_pna_no_rl]
})

# Set the color mapping similar to the previous plot
color_map = ['darkblue', 'lightblue', 'lightgreen']

# Plotting
fig, ax = plt.subplots(figsize=(10, 8))
bars = ax.bar(data['Model'], data['Average Difference'], color=color_map)
ax.set_ylabel('Average Difference')
ax.set_title('Average Difference between Validated Antiviral Drug Targets and All Genes')

# Add value labels on top of the bars
for bar in bars:
    yval = bar.get_height()
    ax.text(bar.get_x() + bar.get_width() / 2, yval, round(yval, 3), va='bottom', ha='center')

# Show plot
plt.show()

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Data
mlp_aupr_results = [0.9363478445208804, 0.890435251895358, 0.7919067920796325, 0.8474321776031543, 0.7568914194124966]
mlp_mcc_results = [0.8367053365673708, 0.7481100033260549, 0.6664971010617428, 0.6797780302145279, 0.5935222088409858]

gcn_aupr_results = [0.9344068891087581, 0.8850526608469108, 0.8982551994909009, 0.8632442520967585, 0.7918964427394495]
gcn_mcc_results = [0.8108728367134042, 0.7311297966919769, 0.7828058504213655, 0.699167869786168, 0.6179178595425098]

pna_aupr_results = [0.965830560256688, 0.9257028301754088, 0.9145742215471117, 0.8983848740374215, 0.7698338583537078]
pna_mcc_results = [0.8908708063747479, 0.825279470542987, 0.8118978309296367, 0.7670705804550759, 0.5937635047564566]

# Create DataFrame
data = pd.DataFrame({
    'Run': ['MLP', 'GCN', 'PNA'],
    'AUPR': [
        np.mean(mlp_aupr_results),
        np.mean(gcn_aupr_results),
        np.mean(pna_aupr_results)
    ],
    'MCC': [
        np.mean(mlp_mcc_results),
        np.mean(gcn_mcc_results),
        np.mean(pna_mcc_results)
    ]
})

data_long = data.melt(id_vars=['Run'], value_vars=['AUPR', 'MCC'], var_name='Metric', value_name='Value')

# Set the color mapping for each metric
color_map = {'AUPR': 'lightblue', 'MCC': 'darkblue'}

n_metrics = data_long['Metric'].nunique()
bar_width = 0.35
index = np.arange(len(data['Run']))

# Error data
error_aupr = [
    np.std(mlp_aupr_results),
    np.std(gcn_aupr_results),
    np.std(pna_aupr_results)
]

error_mcc = [
    np.std(mlp_mcc_results),
    np.std(gcn_mcc_results),
    np.std(pna_mcc_results)
]

# Plot AUPR and MCC
fig, ax1 = plt.subplots(figsize=(12, 8))

metrics1 = ['AUPR', 'MCC']
errors = [error_aupr, error_mcc]

for i, metric in enumerate(metrics1):
    subset = data_long[data_long['Metric'] == metric]
    ax1.bar(index + i * bar_width, subset['Value'], bar_width, yerr=errors[i], label=metric, color=color_map[metric], capsize=5)

ax1.set_ylabel('Metric Values')
ax1.set_title('AUPR and MCC for Disease Genes across Different Models')
ax1.set_xticks(index + bar_width / 2)
ax1.set_xticklabels(data['Run'])
ax1.set_ylim(0.5, 1.0)  # Zoom in on the y-axis to appreciate the differences
ax1.legend()

# Show plots
plt.show()

# %%

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Data
gcn_no_pt_aupr_results = [0.8947247842589933, 0.7356608211634788, 0.8124443321809034, 0.8691416887835589, 0.7781764142304822]
gcn_no_pt_mcc_results = [0.7738996909186401, 0.5956721361748457, 0.677869209325666, 0.7692559848216484, 0.6308961357848648]

gcn_pt_aupr_results = [0.9344068891087581, 0.8850526608469108, 0.8982551994909009, 0.8632442520967585, 0.7918964427394495]
gcn_pt_mcc_results = [0.8108728367134042, 0.7311297966919769, 0.7828058504213655, 0.699167869786168, 0.6179178595425098]

pna_no_pt_aupr_results = [0.8924799982665522, 0.8077549911996086, 0.8911147682203899, 0.8497383061988709, 0.7243812376180062]
pna_no_pt_mcc_results = [0.7620793358294448, 0.7031129788814189, 0.7609937906253444, 0.7384594925022089, 0.570063827765247]

pna_pt_aupr_results = [0.965830560256688, 0.9257028301754088, 0.9145742215471117, 0.8983848740374215, 0.7698338583537078]
pna_pt_mcc_results = [0.8908708063747479, 0.825279470542987, 0.8118978309296367, 0.7670705804550759, 0.5937635047564566]

# Create DataFrame
data = pd.DataFrame({
    'Run': ['GCN (No Pretraining)', 'GCN (With Pretraining)', 'PNA (No Pretraining)', 'PNA (With Pretraining)'],
    'AUPR': [
        np.mean(gcn_no_pt_aupr_results),
        np.mean(gcn_pt_aupr_results),
        np.mean(pna_no_pt_aupr_results),
        np.mean(pna_pt_aupr_results)
    ],
    'MCC': [
        np.mean(gcn_no_pt_mcc_results),
        np.mean(gcn_pt_mcc_results),
        np.mean(pna_no_pt_mcc_results),
        np.mean(pna_pt_mcc_results)
    ]
})

data_long = data.melt(id_vars=['Run'], value_vars=['AUPR', 'MCC'], var_name='Metric', value_name='Value')

# Set the color mapping for each metric
color_map = {
    'AUPR': {'No Pretraining': 'lightblue', 'Pretraining': 'blue'},
    'MCC': {'No Pretraining': 'lightgreen', 'Pretraining': 'green'}
}

n_metrics = data_long['Metric'].nunique()
bar_width = 0.35
index = np.arange(len(data['Run']))

# Error data
error_aupr = [
    np.std(gcn_no_pt_aupr_results),
    np.std(gcn_pt_aupr_results),
    np.std(pna_no_pt_aupr_results),
    np.std(pna_pt_aupr_results)
]

error_mcc = [
    np.std(gcn_no_pt_mcc_results),
    np.std(gcn_pt_mcc_results),
    np.std(pna_no_pt_mcc_results),
    np.std(pna_pt_mcc_results)
]

# Determine which bar color to use based on the label
def get_color(metric, label):
    if 'No Pretraining' in label:
        return color_map[metric]['No Pretraining']
    else:
        return color_map[metric]['Pretraining']

# Plot AUPR and MCC
fig, ax1 = plt.subplots(figsize=(12, 8))

metrics1 = ['AUPR', 'MCC']
errors = [error_aupr, error_mcc]

for i, metric in enumerate(metrics1):
    subset = data_long[data_long['Metric'] == metric]
    colors = [get_color(metric, label) for label in subset['Run']]
    ax1.bar(index + i * bar_width, subset['Value'], bar_width, yerr=errors[i], label=metric, color=colors, capsize=5)

ax1.set_ylabel('Metric Values')
ax1.set_title('AUPR and MCC for GCN and PNA Models with and without Pretraining')
ax1.set_xticks(index + bar_width / 2)
ax1.set_xticklabels(data['Run'])
ax1.set_ylim(0.5, 1.0)  # Zoom in on the y-axis to appreciate the differences
ax1.legend()

# Show plots
plt.show()

 # %%

 import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Data
gcn_no_dt_aupr_results = [0.779573306569122, 0.7229010525112871, 0.7903724391499631, 0.4824945220645338, 0.6029065601478663]
gcn_no_dt_mcc_results = [0.6522734525161044, 0.6080082478589438, 0.6209511540958679, 0.26926960044597725, 0.4639505985068466]

gcn_dt_aupr_results = [0.9344068891087581, 0.8850526608469108, 0.8982551994909009, 0.8632442520967585, 0.7918964427394495]
gcn_dt_mcc_results = [0.8108728367134042, 0.7311297966919769, 0.7828058504213655, 0.699167869786168, 0.6179178595425098]

pna_no_dt_aupr_results = [0.9485376097893509, 0.7779478231441725, 0.890006244932742, 0.8733515513502285, 0.6961124805834528]
pna_no_dt_mcc_results = [0.8559149175796082, 0.630659322894917, 0.7475520779743993, 0.7248376292417312, 0.5525339444335872]

pna_dt_aupr_results = [0.9661826019641085, 0.9372626083122467, 0.9090680777369204, 0.8858430969426251, 0.7688624177550732]
pna_dt_mcc_results = [0.8989694666094653, 0.8341115254075708, 0.7785795543414196, 0.770708262818106, 0.5719694409972929]

# Create DataFrame
data = pd.DataFrame({
    'Run': ['GCN (No DTs in PT)', 'GCN (DTs in PT)', 'PNA (No DTs in PT)', 'PNA (DTs in PT)'],
    'AUPR': [
        np.mean(gcn_no_dt_aupr_results),
        np.mean(gcn_dt_aupr_results),
        np.mean(pna_no_dt_aupr_results),
        np.mean(pna_dt_aupr_results)
    ],
    'MCC': [
        np.mean(gcn_no_dt_mcc_results),
        np.mean(gcn_dt_mcc_results),
        np.mean(pna_no_dt_mcc_results),
        np.mean(pna_dt_mcc_results)
    ]
})

data_long = data.melt(id_vars=['Run'], value_vars=['AUPR', 'MCC'], var_name='Metric', value_name='Value')

# Set the color mapping for each metric
color_map = {
    'AUPR': {'No DTs in PT': 'lightblue', 'DTs in PT': 'blue'},
    'MCC': {'No DTs in PT': 'lightgreen', 'DTs in PT': 'green'}
}

n_metrics = data_long['Metric'].nunique()
bar_width = 0.35
index = np.arange(len(data['Run']))

# Error data
error_aupr = [
    np.std(gcn_no_dt_aupr_results),
    np.std(gcn_dt_aupr_results),
    np.std(pna_no_dt_aupr_results),
    np.std(pna_dt_aupr_results)
]

error_mcc = [
    np.std(gcn_no_dt_mcc_results),
    np.std(gcn_dt_mcc_results),
    np.std(pna_no_dt_mcc_results),
    np.std(pna_dt_mcc_results)
]

# Determine which bar color to use based on the label
def get_color(metric, label):
    if 'No DTs in PT' in label:
        return color_map[metric]['No DTs in PT']
    else:
        return color_map[metric]['DTs in PT']

# Plot AUPR and MCC
fig, ax1 = plt.subplots(figsize=(12, 8))

metrics1 = ['AUPR', 'MCC']
errors = [error_aupr, error_mcc]

for i, metric in enumerate(metrics1):
    subset = data_long[data_long['Metric'] == metric]
    colors = [get_color(metric, label) for label in subset['Run']]
    ax1.bar(index + i * bar_width, subset['Value'], bar_width, yerr=errors[i], label=metric, color=colors, capsize=5)

ax1.set_ylabel('Metric Values')
ax1.set_title('AUPR and MCC for GCN and PNA Models with and without Drug Targets (DTs) in Pre-Training (PT)')
ax1.set_xticks(index + bar_width / 2)
ax1.set_xticklabels(data['Run'])
ax1.set_ylim(0.2, 1.0)  # Adjusted y-axis to start from 0.2
ax1.legend()

# Show plots
plt.show()

# %%

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Data
mlp_rand_aupr_results = [0.31883233835772196, 0.29190745901536114, 0.28113631036124914, 0.2503798680040111, 0.40069236822667703]
mlp_rand_mcc_results = [0.040990047799970626, 0.08868556743632698, 0.10988167744215564, -0.01027222585602276, 0.2792635601192868]

gcn_rand_aupr_results = [0.5845362144130383, 0.602479796984503, 0.7479875905528002, 0.8079970768356918, 0.5281653471906163]
gcn_rand_mcc_results = [0.4944410369195648, 0.5218702354139774, 0.6312552017000902, 0.6936876838886428, 0.434360233389433]

gcn_aupr_results = [0.9344068891087581, 0.8850526608469108, 0.8982551994909009, 0.8632442520967585, 0.7918964427394495]
gcn_mcc_results = [0.8108728367134042, 0.7311297966919769, 0.7828058504213655, 0.699167869786168, 0.6179178595425098]

pna_rand_aupr_results = [0.8410623158589473, 0.7550991395132985, 0.8189776955547418, 0.8469894766046352, 0.7127875169633725]
pna_rand_mcc_results = [0.6841849821211788, 0.6065260455264206, 0.6979789395364095, 0.7507849268443677, 0.5246734154940326]

pna_aupr_results = [0.965830560256688, 0.9257028301754088, 0.9145742215471117, 0.8983848740374215, 0.7698338583537078]
pna_mcc_results = [0.8908708063747479, 0.825279470542987, 0.8118978309296367, 0.7670705804550759, 0.5937635047564566]

# Create DataFrame
data = pd.DataFrame({
    'Run': ['MLP (Rand Features)', 'GCN (Rand Features)', 'GCN', 'PNA (Rand Features)', 'PNA'],
    'AUPR': [
        np.mean(mlp_rand_aupr_results),
        np.mean(gcn_rand_aupr_results),
        np.mean(gcn_aupr_results),
        np.mean(pna_rand_aupr_results),
        np.mean(pna_aupr_results)
    ],
    'MCC': [
        np.mean(mlp_rand_mcc_results),
        np.mean(gcn_rand_mcc_results),
        np.mean(gcn_mcc_results),
        np.mean(pna_rand_mcc_results),
        np.mean(pna_mcc_results)
    ]
})

data_long = data.melt(id_vars=['Run'], value_vars=['AUPR', 'MCC'], var_name='Metric', value_name='Value')

# Set the color mapping for each metric
color_map = {
    'AUPR': {'Rand Features': 'lightblue', 'Normal': 'blue'},
    'MCC': {'Rand Features': 'lightgreen', 'Normal': 'green'}
}

n_metrics = data_long['Metric'].nunique()
bar_width = 0.35
index = np.arange(len(data['Run']))

# Error data
error_aupr = [
    np.std(mlp_rand_aupr_results),
    np.std(gcn_rand_aupr_results),
    np.std(gcn_aupr_results),
    np.std(pna_rand_aupr_results),
    np.std(pna_aupr_results)
]

error_mcc = [
    np.std(mlp_rand_mcc_results),
    np.std(gcn_rand_mcc_results),
    np.std(gcn_mcc_results),
    np.std(pna_rand_mcc_results),
    np.std(pna_mcc_results)
]

# Determine which bar color to use based on the label
def get_color(metric, label):
    if 'Rand Features' in label:
        return color_map[metric]['Rand Features']
    else:
        return color_map[metric]['Normal']

# Plot AUPR and MCC
fig, ax1 = plt.subplots(figsize=(12, 8))

metrics1 = ['AUPR', 'MCC']
errors = [error_aupr, error_mcc]

for i, metric in enumerate(metrics1):
    subset = data_long[data_long['Metric'] == metric]
    colors = [get_color(metric, label) for label in subset['Run']]
    ax1.bar(index + i * bar_width, subset['Value'], bar_width, yerr=errors[i], label=metric, color=colors, capsize=5)

ax1.set_ylabel('Metric Values')
ax1.set_title('AUPR and MCC for MLP, GCN, and PNA Models with and without Random Features')
ax1.set_xticks(index + bar_width / 2)
ax1.set_xticklabels(data['Run'])
ax1.set_ylim(0, 1.0)  # Adjusted y-axis to start from 0
ax1.legend()

# Show plots
plt.show()


# %%

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Data
pna_viral_aupr_results = [0.8648813327145825, 0.8918708848538781, 0.8696527821456964, 0.8676331482264854, 0.7402539731874734]
pna_viral_mcc_results = [0.8016127461171916, 0.7566332503066436, 0.7113670216587815, 0.7135171259934024, 0.5628802873099225]

pna_all_aupr_results = [0.965830560256688, 0.9257028301754088, 0.9145742215471117, 0.8983848740374215, 0.7698338583537078]
pna_all_mcc_results = [0.8908708063747479, 0.825279470542987, 0.8118978309296367, 0.7670705804550759, 0.5937635047564566]

# Create DataFrame
data = pd.DataFrame({
    'Run': ['PNA (Viral Diseases PT)', 'PNA (All Diseases PT)'],
    'AUPR': [
        np.mean(pna_viral_aupr_results),
        np.mean(pna_all_aupr_results)
    ],
    'MCC': [
        np.mean(pna_viral_mcc_results),
        np.mean(pna_all_mcc_results)
    ]
})

data_long = data.melt(id_vars=['Run'], value_vars=['AUPR', 'MCC'], var_name='Metric', value_name='Value')

# Set the color mapping for each metric
color_map = {
    'AUPR': {'Viral Diseases PT': 'lightblue', 'All Diseases PT': 'blue'},
    'MCC': {'Viral Diseases PT': 'lightgreen', 'All Diseases PT': 'green'}
}

n_metrics = data_long['Metric'].nunique()
bar_width = 0.35
index = np.arange(len(data['Run']))

# Error data
error_aupr = [
    np.std(pna_viral_aupr_results),
    np.std(pna_all_aupr_results)
]

error_mcc = [
    np.std(pna_viral_mcc_results),
    np.std(pna_all_mcc_results)
]

# Determine which bar color to use based on the label
def get_color(metric, label):
    if 'Viral Diseases PT' in label:
        return color_map[metric]['Viral Diseases PT']
    else:
        return color_map[metric]['All Diseases PT']

# Plot AUPR and MCC
fig, ax1 = plt.subplots(figsize=(12, 8))

metrics1 = ['AUPR', 'MCC']
errors = [error_aupr, error_mcc]

for i, metric in enumerate(metrics1):
    subset = data_long[data_long['Metric'] == metric]
    colors = [get_color(metric, label) for label in subset['Run']]
    ax1.bar(index + i * bar_width, subset['Value'], bar_width, yerr=errors[i], label=metric, color=colors, capsize=5)

ax1.set_ylabel('Metric Values')
ax1.set_title('AUPR and MCC for PNA Models with Pre-training on Viral Diseases and All Diseases')
ax1.set_xticks(index + bar_width / 2)
ax1.set_xticklabels(data['Run'])
ax1.set_ylim(0.5, 1.0)  # Adjusted y-axis to start from 0
ax1.legend()

# Show plots
plt.show()

# %%

import pandas as pd
import matplotlib.pyplot as plt
from pandas.plotting import table

# Data for the table
data = {
    'Training Data': [
        'Total Feature Vector Size per Protein', 
        'PPI Positional Encoding dimension', 
        'GO Functional Embedding dimension', 
        'ESM2 LLM Embedding dimension', 
        'Multi-omics input feature dimension', 
        'PPI Network # Genes', 
        'PPI Network # Connections', 
        'Pre-training Disease genes classification: Positives', 
        'Pre-training Disease genes classification: Negatives', 
        'Pre-training Disease genes classification: Unlabeled',
        'Pre-training Drug Targets classification: Positives', 
        'Pre-training Drug Targets classification: Negatives',
        'Pre-training Drug Targets classification: Unlabeled', 
        'Fine-tuning Host Factors specific to Sars-CoV-2 Positives', 
        'Fine-tuning Host Factors specific to Sars-CoV-2 Negatives',
        'Fine-tuning Host Factors specific to Sars-CoV-2 Unlabeled'
    ],
    '#': [
        '790', 
        '128', 
        '128', 
        '480', 
        '54', 
        '18527', 
        '881166', 
        '5870', 
        '5349', 
        '7308',
        '1495', 
        '5843', 
        '11189',
        '190', 
        '6631',
        '11706'
    ]
}

# Create DataFrame
df = pd.DataFrame(data)

# Plot the table
fig, ax = plt.subplots(figsize=(12, 8))  # Set size frame
ax.axis('tight')
ax.axis('off')
tab = table(ax, df, loc='center', cellLoc='center', colWidths=[0.3, 0.3])

tab.auto_set_font_size(False)
tab.set_fontsize(14)
tab.scale(1.5, 1.5)  # Scale the table size

# Style adjustments
tab.auto_set_column_width([0, 1])  # Adjust column width to fit text

# Set a nicer looking font and style
plt.rcParams.update({'font.family': 'Arial', 'font.size': 12})

# Show the plot
plt.show()


# %%

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Data
gcn_expl_aupr_results = [0.9423114766271445, 0.8940694803232807, 0.8793061581585824, 0.8460160653271597, 0.7905641832322687]
gcn_expl_mcc_results = [0.8325510990961368, 0.7797119523194819, 0.7381824110376239, 0.7159403956766562, 0.5593110523767693]

gcn_aupr_results = [0.9344068891087581, 0.8850526608469108, 0.8982551994909009, 0.8632442520967585, 0.7918964427394495]
gcn_mcc_results = [0.8108728367134042, 0.7311297966919769, 0.7828058504213655, 0.699167869786168, 0.6179178595425098]

pna_expl_aupr_results = [0.9445126996513716, 0.9249112399575523, 0.9063438542647333, 0.8917603340454718, 0.7482304946480477]
pna_expl_mcc_results = [0.8250022449650939, 0.8261020773743314, 0.7872284709501377, 0.7965495076760088, 0.5812429656868048]

pna_aupr_results = [0.9613328417481091, 0.9254784690976156, 0.9074067494043563, 0.8928765481636236, 0.7340699847509894]
pna_mcc_results = [0.8927286368750905, 0.8181160591575534, 0.7715456992467483, 0.7599210540633274, 0.5192514889462698]

# Create DataFrame
data = pd.DataFrame({
    'Run': ['GCN (Expl Feedback)', 'GCN', 'PNA (Expl Feedback)', 'PNA'],
    'AUPR': [
        np.mean(gcn_expl_aupr_results),
        np.mean(gcn_aupr_results),
        np.mean(pna_expl_aupr_results),
        np.mean(pna_aupr_results)
    ],
    'MCC': [
        np.mean(gcn_expl_mcc_results),
        np.mean(gcn_mcc_results),
        np.mean(pna_expl_mcc_results),
        np.mean(pna_mcc_results)
    ]
})

data_long = data.melt(id_vars=['Run'], value_vars=['AUPR', 'MCC'], var_name='Metric', value_name='Value')

# Set the color mapping for each metric
color_map = {
    'AUPR': {'Expl Feedback': 'lightblue', 'Normal': 'blue'},
    'MCC': {'Expl Feedback': 'lightgreen', 'Normal': 'green'}
}

n_metrics = data_long['Metric'].nunique()
bar_width = 0.35
index = np.arange(len(data['Run']))

# Error data
error_aupr = [
    np.std(gcn_expl_aupr_results),
    np.std(gcn_aupr_results),
    np.std(pna_expl_aupr_results),
    np.std(pna_aupr_results)
]

error_mcc = [
    np.std(gcn_expl_mcc_results),
    np.std(gcn_mcc_results),
    np.std(pna_expl_mcc_results),
    np.std(pna_mcc_results)
]

# Determine which bar color to use based on the label
def get_color(metric, label):
    if 'Expl Feedback' in label:
        return color_map[metric]['Expl Feedback']
    else:
        return color_map[metric]['Normal']

# Plot AUPR and MCC
fig, ax1 = plt.subplots(figsize=(12, 8))

metrics1 = ['AUPR', 'MCC']
errors = [error_aupr, error_mcc]

for i, metric in enumerate(metrics1):
    subset = data_long[data_long['Metric'] == metric]
    colors = [get_color(metric, label) for label in subset['Run']]
    ax1.bar(index + i * bar_width, subset['Value'], bar_width, yerr=errors[i], label=metric, color=colors, capsize=5)

ax1.set_ylabel('Metric Values')
ax1.set_title('AUPR and MCC for GCN and PNA Models with and without Explanation Feedback')
ax1.set_xticks(index + bar_width / 2)
ax1.set_xticklabels(data['Run'])
ax1.set_ylim(0.6, 1.0)  # Adjusted y-axis to start from 0.5 to 1.0
ax1.legend()

# Show plots
plt.show()

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Data
pna_rand_aupr_results = [0.8410623158589473, 0.7550991395132985, 0.8189776955547418, 0.8469894766046352, 0.7127875169633725]
pna_rand_mcc_results = [0.6841849821211788, 0.6065260455264206, 0.6979789395364095, 0.7507849268443677, 0.5246734154940326]

pna_no_pos_aupr_results = [0.9108103847644654, 0.8319877153803765, 0.9076022918643095, 0.866885721499041, 0.7578791788561539]
pna_no_pos_mcc_results = [0.793877897802164, 0.7066528178044255, 0.7764476266134599, 0.7297945833172871, 0.6176628372538638]

pna_no_omics_aupr_results = [0.9445714061129551, 0.8770200279044603, 0.8831774981853341, 0.8893422094981327, 0.7568012322354125] 
pna_no_omics_mcc_results = [0.8068611283870922, 0.7435208654681394, 0.770922452016368, 0.7478628389293359, 0.5515714710687208]

pna_no_go_aupr_results = [0.9661826019641085, 0.9372626083122467, 0.9090680777369204, 0.8858430969426251, 0.7688624177550732]
pna_no_go_mcc_results = [0.8989694666094653, 0.8341115254075708, 0.7785795543414196, 0.770708262818106, 0.5719694409972929]

pna_no_prot_emb_aupr_results = [0.9555539505349887, 0.9286501767540445, 0.906804050080439, 0.8955905316754332, 0.7631950126512196]
pna_no_prot_emb_mcc_results = [0.8712464352890902, 0.8201108147635243, 0.7859274577555407, 0.789445809638981, 0.5594433623167769]

pna_all_aupr_results = [0.9640936241847982, 0.9273033186742462, 0.9182875817885985, 0.9035288598829728, 0.7647945459797746]
pna_all_mcc_results = [0.876504225327988, 0.8427900175243171, 0.8190168001267907, 0.7521427011765147, 0.6017023508807972]

# Create DataFrame
data = pd.DataFrame({
    'Run': [
        'PNA (Rand Features)', 
        'PNA (No Pos Encoding)', 
        'PNA (No OMICS)',  
        'PNA (No GO)', 
        'PNA (No Prot Emb)', 
        'PNA (All Inputs)'
    ],
    'AUPR': [
        np.mean(pna_rand_aupr_results),
        np.mean(pna_no_pos_aupr_results),
        np.mean(pna_no_omics_aupr_results),
        np.mean(pna_no_go_aupr_results),
        np.mean(pna_no_prot_emb_aupr_results),
        np.mean(pna_all_aupr_results)
    ],
    'MCC': [
        np.mean(pna_rand_mcc_results),
        np.mean(pna_no_pos_mcc_results),
        np.mean(pna_no_omics_mcc_results),
        np.mean(pna_no_go_mcc_results),
        np.mean(pna_no_prot_emb_mcc_results),
        np.mean(pna_all_mcc_results)
    ]
})

data_long = data.melt(id_vars=['Run'], value_vars=['AUPR', 'MCC'], var_name='Metric', value_name='Value')

# Set the color mapping for each metric
color_map = {
    'AUPR': {'Rand Features': 'lightblue', 'No Pos Encoding': 'lightblue', 'No Omics': 'lightblue', 'No GO': 'lightblue', 'No Prot Emb': 'lightblue', 'All Inputs': 'lightblue'},
    'MCC': {'Rand Features': 'lightseagreen', 'No Pos Encoding': 'lightseagreen', 'No Omics': 'lightseagreen','No GO': 'lightseagreen', 'No Prot Emb': 'lightseagreen', 'All Inputs': 'lightseagreen'}
}

n_metrics = data_long['Metric'].nunique()
bar_width = 0.35
index = np.arange(len(data['Run']))

# Error data
error_aupr = [
    np.std(pna_rand_aupr_results),
    np.std(pna_no_pos_aupr_results),
    np.std(pna_no_omics_aupr_results),
    np.std(pna_no_go_aupr_results),
    np.std(pna_no_prot_emb_aupr_results),
    np.std(pna_all_aupr_results)
]

error_mcc = [
    np.std(pna_rand_mcc_results),
    np.std(pna_no_pos_mcc_results),
    np.std(pna_no_omics_mcc_results),
    np.std(pna_no_go_mcc_results),
    np.std(pna_no_prot_emb_mcc_results),
    np.std(pna_all_mcc_results)
]

# Determine which bar color to use based on the label
def get_color(metric, label):
    if 'Rand Features' in label:
        return color_map[metric]['Rand Features']
    elif 'No Pos Encoding' in label:
        return color_map[metric]['No Pos Encoding']
    elif 'No Omics' in label:
        return color_map[metric]['No Omics']
    elif 'No GO' in label:
        return color_map[metric]['No GO']
    elif 'No Prot Emb' in label:
        return color_map[metric]['No Prot Emb']
    else:
        return color_map[metric]['All Inputs']

# Plot AUPR and MCC
fig, ax1 = plt.subplots(figsize=(12, 8))

metrics1 = ['AUPR', 'MCC']
errors = [error_aupr, error_mcc]

for i, metric in enumerate(metrics1):
    subset = data_long[data_long['Metric'] == metric]
    colors = [get_color(metric, label) for label in subset['Run']]
    ax1.bar(index + i * bar_width, subset['Value'], bar_width, yerr=errors[i], label=metric, color=colors, capsize=5)

ax1.set_ylabel('Metric Values')
ax1.set_title('AUPR and MCC for PNA Models with Varying Inputs')
ax1.set_xticks(index + bar_width / 2)
ax1.set_xticklabels(data['Run'], rotation=45, ha='right')
ax1.set_ylim(0.55, 1.0)  # Adjusted y-axis to start from 0.5 to 1.0
ax1.legend()

# Show plots
plt.show()

# %%

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Data
data = {
    'Model': [
        'PNA (Rand Features)', 'PNA', 'PNA (No DT during PT)', 'PNA (Guided DT Learning)', 
        'GCN (Rand Features)', 'GCN'
    ],
    'HF Classified as DG Fraction': [
        0.5412413954734803, 0.5144827783107757, 0.4293990254402161, 0.614660120010376, 
        0.6257340133190155, 0.5534581422805787
    ],
    'Inferred DT Classified as DG Fraction': [
        0.87027028799057, 0.8216216444969178, 0.7081081151962281, 0.8648648858070374, 
        0.8972973227500916, 0.8486486673355103
    ],
    'Validated DT Classified as DG Fraction': [
        0.6078431606292725, 0.41568629145622255, 0.27058824598789216, 0.5244444608688354, 
        0.7058823704719543, 0.4431372702121735
    ]
}

# Create DataFrame
df = pd.DataFrame(data)

# Plot
fig, ax = plt.subplots(figsize=(14, 8))

bar_width = 0.2
index = np.arange(len(df['Model']))

# Plot each metric
ax.bar(index - bar_width, df['HF Classified as DG Fraction'], bar_width, label='HF Classified as DG Fraction', color='lightblue')
ax.bar(index, df['Inferred DT Classified as DG Fraction'], bar_width, label='Inferred DT Classified as DG Fraction', color='blue')
ax.bar(index + bar_width, df['Validated DT Classified as DG Fraction'], bar_width, label='Validated DT Classified as DG Fraction', color='darkblue')

# Customize the plot
ax.set_xlabel('Model')
ax.set_ylabel('Fraction')
ax.set_title('Comparison of PNA and GCN Models on Validation Data')
ax.set_xticks(index)
ax.set_xticklabels(df['Model'], rotation=45, ha='right')
ax.set_ylim(0, 1.0)  # Set y-axis range from 0 to 1
ax.legend()

# Show plot
plt.tight_layout()
plt.show()


# %%
