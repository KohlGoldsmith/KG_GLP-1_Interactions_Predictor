# GLP-1 Drug-Drug Interaction Predictor
## Kohl Goldsmith for MTU SAT5141 Clinical Decision Support & AI Modelling

### Purpose
  The idea of this project is to provide AI driven predictions in drug-drug interactions. Specifically involving the more frequently used and novel GLP-1 drugs like semaglutide (Ozempic, Wegovy, Rybelsus), 
  dulaglutide (Trulicity), liraglutide (Victoza, Saxenda), exenatide (Byetta, Bydureon), and tirzepatide (Mounjaro, Zepbound).

### Configuration / Setup
  Copy the directory schema of the Github through cloning or manual setup.

### Running the main file for date preprocessing
  Run main.py, watching for errors in saving, data is recommended to come from 2024 or newer.
  Once this is completed, real analysis can be done.

  ### Next step, training.
  Once data has successfully been preprocessed into a faers_with_embeddings_ready.csv file, run the rf_model_test.py in the testing directory to perform a random forest training sesion on the data.
  
