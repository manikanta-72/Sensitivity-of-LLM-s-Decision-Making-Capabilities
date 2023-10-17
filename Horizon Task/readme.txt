This folder contains the code to reproduce our experiments. The code has been adapted from https://github.com/marcelbinz/GPT3goesPsychology
There are 3 subfolders one for each model.
Each folder contains python files(refer paper(Figure 2) for prompt information):
    1. original_prompt.py -- contains the code that queries LLM with unchanged prompt from Binz and Schulz (2023)
    2. quasi-CoT.py -- contains code with quasi-CoT prompt
    3. CoT.py -- contains code with CoT prompt
    4. CoT-Exploit.py -- contains code with CoT-Exploit prompt
    5. CoT-Explore.py -- contains code with CoT-Explore prompt
To obtain the results for "Varying Temperature" section, Change the temperature in line 28 of original_prompt.py

To run any of the file enter the following command in the terminal
python FILE_NAME.py

To visualize the data use figures.ipynb

All the code, experiment and data will be made public upon acceptance