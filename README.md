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

  ### Using processed data and running models
  Once data has successfully been preprocessed into a faers_with_embeddings_ready.csv file, run the rf_model_test.py in the testing directory to perform a random forest training sesion on the data.
  
<img width="311" height="403" alt="Screenshot 2025-11-19 at 9 58 00 AM" src="https://github.com/user-attachments/assets/143ee652-22bc-4654-ad10-18757721ac50" />

Use the setting above within 'rf_model_test.py' to set testing parameters. This should be possible to implement on a web front end if everything funtions correctly.
