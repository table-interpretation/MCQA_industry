## Multi-choice question answering (MCQA) dataset for eliciting engineering knowledge from industrial knowledge graph

In the folder [data](data) we provide a synthetically generated dataset simulating real-world engineering data. 
The test and train tables contain information about equipments, their quantities and unit of measurements. 


The table below is an example of the first 3 rows from the test dataframe. In the column _Question_ is the prompt 
that we use to query the LLM. The list of extracted candidates is in the column final choices. If the number of choices is less than 5, the first Quant is used for additional semantic search. The column
label indicates the index of the correct answer.

| **Eq_Label**               | **Quant**                | **UOM**      | **Datum**  | **Question**                                                                                     | final_choices                                                                                               | label |  
|------------------------|----------------------|----------|--------|----------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------|-------|   
| filter residue pump    | max design pressure  | kbar     | 4.6    | Which of the following refers to the 4.6 kbar of a filter residue pump?                      | ['design pressure', 'allowable loading pressure', **'max design pressure'**, 'wind pressure']               | 2     |  
| radial bearing         | max axial load       | kdyn     | 86.48  | Which of the following refers to the 86.48 kdyn of a radial bearing?                         | ['brake force', 'wind load', **'max axial load'**, 'support load']                                          | 2     |  
| bladder accumulator    | inlet pressure       | cmatdeg  | 66.0   | Which of the following refers to the 66.0 cmatdeg of a bladder accumulator?                  | ['allowable loading pressure', 'design pressure', 'max design pressure', 'hydrostatic test pressure', **'inlet pressure'**] | 4     |

The script [gpt_inference](gpt_inference.py) can be used for evaluation of LLMs on the MCQA task. 

The folder [preprocess](preprocess) contains the scripts used for creating the KG and for the selection of the candidates.