import matplotlib.pyplot as plt
import numpy as np
import matplotlib.dates as mdates
from datetime import datetime

plt.rcParams.update({
    "lines.color": "white",
    "patch.edgecolor": "white",
    "text.color": "white",
    "axes.facecolor": "black",
    "axes.edgecolor": "lightgray",
    "axes.labelcolor": "white",
    "xtick.color": "white",
    "ytick.color": "white",
    "grid.color": "lightgray",
    "figure.facecolor": "black",
    "figure.edgecolor": "black",
    "savefig.facecolor": "black",
    "savefig.edgecolor": "black"})

# date, billions of tokens, billions of parameters
# Define the data (excluding models with N/A values)
models = {
    'BERT Large': ('31 October 2018', 3.3, 0.34, 'https://arxiv.org/abs/1810.04805', 'https://github.com/google-research/bert', 'Google AI'),
    'GPT-2': ('14 February 2019', 10, 1.542, 'https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf','https://openai.com/index/better-language-models/', 'OpenAI'),
    'RoBERTa': ('26 July 2019', 40, 0.340, 'https://arxiv.org/abs/1907.11692', 'https://github.com/facebookresearch/fairseq/blob/main/examples/roberta/README.md', 'University of Washington and Facebook'),
    'Megatron-LM': ('17 September 2019', 43.5, 8.3, 'https://arxiv.org/abs/1909.08053', 'https://github.com/NVIDIA/Megatron-LM', 'NVIDIA'),
    'Flan T5-11B': ('23 October 2019', 187.5, 11, 'https://arxiv.org/pdf/2210.11416', 'https://research.google/blog/exploring-transfer-learning-with-t5-the-text-to-text-transfer-transformer/', 'Google'),
    'GPT-3': ('22 July 2020', 400, 175, 'https://arxiv.org/pdf/2005.14165', '', 'OpenAI'),
    'Megatron-Turing NLG': ('28 January 2022', 339, 530, 'https://arxiv.org/abs/2201.11990', 'https://developer.nvidia.com/blog/using-deepspeed-and-megatron-to-train-megatron-turing-nlg-530b-the-worlds-largest-and-most-powerful-generative-language-model/', 'NVIDIA'),
    'Gopher': ('8 December 2021', 300, 280, 'https://arxiv.org/pdf/2112.11446', 'https://deepmind.google/discover/blog/language-modelling-at-scale-gopher-ethical-considerations-and-retrieval/', 'Deepmind'),
    'OPT-175B': ('2 May 2022', 180, 175, 'https://arxiv.org/abs/2205.01068', 'https://ai.meta.com/blog/democratizing-access-to-large-scale-language-models-with-opt-175b/', 'Meta'),
    'PaLM': ('5 October 2022', 780, 540, 'https://arxiv.org/abs/2204.02311', 'https://ai.google/get-started/our-models/', 'Google Research'),
    'BLOOM': ('9 November 2022', 366, 176, 'https://arxiv.org/abs/2211.05100', '', 'Big Science Workshop'),
    'LLaMA 1': ('27 February 2023', 1400, 65, 'https://arxiv.org/abs/2302.13971', 'https://ai.meta.com/blog/large-language-model-llama-meta-ai/', 'Meta AI'),
    'GPT-4' : ('15 March 2023',13000,1000, 'https://arxiv.org/pdf/2303.08774', 'https://openai.com/index/gpt-4/', 'OpenAI'),
    'LLaMA 2': ('18 July 2023', 2000, 70, 'https://arxiv.org/abs/2307.09288', 'https://ai.meta.com/research/publications/llama-2-open-foundation-and-fine-tuned-chat-models/', 'GenAI, Meta'),
    'Claude 3 Opus' : ('4 March 2024', 40000, 2000, '', 'https://www.anthropic.com/news/claude-3-family', 'Anthropic'),
    'Gemma 2-27B' : ('31 July 2024',13000,27, 'https://arxiv.org/abs/2408.00118', 'https://ai.google.dev/gemma/docs/core/model_card_2', 'Google AI'),
    'LLaMA 3-405B' : ('31 July 2024', 16550, 405, 'https://arxiv.org/abs/2407.21783', 'https://ai.meta.com/blog/meta-llama-3/', 'Meta AI'),
    'GPT-5' : ('7 August 2025', 114000, 5000, 'https://lifearchitect.ai/gpt-5/', 'https://openai.com/index/introducing-gpt-5/', 'OpenAI')
}

# Extract data and convert dates to datetime objects
dates = []
tokens = []
params = []
papers = []
websites = []
organisations = []
names = []

for model, data in models.items():
    date_str = data[0]
    date_obj = datetime.strptime(date_str, '%d %B %Y')
    
    dates.append(date_obj)
    tokens.append(data[1])
    params.append(data[2])
    papers.append(data[3])
    websites.append(data[4])
    organisations.append(data[5])
    names.append(model)

# Create the scatter plot
plt.figure(figsize=(19.2, 10.8))
plt.style.use('dark_background')

# Scale the sizes to be visible but not overwhelming
sizes = [p * 5 for p in params]  # Multiply by 5 to make sizes more visible

# Create scatter plot with datetime objects
scatter = plt.scatter(dates, tokens, s=sizes, alpha=0.9, c=params, 
                     cmap='plasma', edgecolor='white', linewidth=1)

# Add labels for each point with line break between name and organization
# Position labels based on parameter count - higher for larger bubbles
for i, name in enumerate(names):
    # Create label with line break
    label = f"{name}"
    
    # Calculate vertical offset based on parameter count
    # Base offset of 10 pixels plus additional offset scaled by parameter count
    # Taking the square root to avoid extreme offsets for very large parameter values
    vertical_offset = 7 + np.sqrt(params[i]) * 1.0
    
    # Add annotation with dynamic vertical positioning
    plt.annotate(label, (dates[i], tokens[i]), 
                xytext=(0, vertical_offset),  # 0 x offset (centered), dynamic y offset
                textcoords='offset points',
                fontsize=8,
                ha='center',  # Horizontal alignment: center
                va='bottom')  # Vertical alignment: bottom

# Customize the plot
plt.title('LLM Models: # of Training Tokens vs Release Year\n(Bubble size represents parameter count)',
          fontsize=24, pad=20, fontfamily='Public Sans')
plt.xlabel('Release Date', fontfamily='Public Sans', size=18)
plt.ylabel('Training Tokens (Billions)', fontfamily='Public Sans', size=18)

# Format the x-axis to use the desired date format
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%d %B %Y'))

# Set the x-axis to display nicely spaced dates
plt.gca().xaxis.set_major_locator(mdates.YearLocator())
plt.gcf().autofmt_xdate(rotation=45)  # Rotate date labels for better readability

# Add a colorbar to show parameter scale
cbar = plt.colorbar(scatter)
cbar.set_label('Parameters (Billions)', rotation=270, labelpad=15, fontfamily='Public Sans', size=18)

# Adjust axes
plt.grid(True, linestyle='--', alpha=0.7)
plt.axhline(y=0, color='yellow', linestyle='-', linewidth=0.5, alpha=0.6)

# Axis limits 
ax = plt.gca()
ax.set_xlim([datetime(2018, 1, 1), datetime(2027, 1, 1)])
ax.set_ylim([-15000, 150000])       # Instead of [-10000, 200000]      

# Adjust layout to prevent label cutoff
plt.tight_layout(pad=7)
plt.savefig('training-tokens.png')
plt.show()
