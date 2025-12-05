# GLP-1 Drug-Drug Interaction Predictor
## Kohl Goldsmith for MTU SAT5141 Clinical Decision Support & AI Modelling

### Purpose
The idea of this project is to provide AI driven predictions in drug-drug interactions. Specifically involving the more frequently used and novel GLP-1 drugs like semaglutide (Ozempic, Wegovy, Rybelsus), dulaglutide (Trulicity), liraglutide (Victoza, Saxenda), exenatide (Byetta, Bydureon), and tirzepatide (Mounjaro, Zepbound). Using the dictionaries defined in the HODDI dataset [10] were essential for the results of this model.

### Configuration / Setup
  **Clone the Repository**: Copy the directory schema of the Github through cloning or manual setup.

  **Install Dependencies**: Run pip install -r requirements.txt to ensure the packages are installed onto your system.

  **Download Missing Data**: You WILL be missing the Unique_Side_Effect.csv needed to run the preprocessing stages.
    
- Download it here: TIML-Group/HODDI/Side_effects_unique.csv
* Place it in your /dictionary directory

### Running the main file for date preprocessing
Run main.py, watching for errors in saving, data is recommended to come from 2024 or newer.
Once this is completed, analysis and training operations can begin.

#### Setting Parameters
- In model_test.py, line 155-172, the following arguments will be manually changed to test different subjects.
- user_symptoms = ["Abdominal pain", "Nausea", "Gastrointestinal disorder"] : Change these to match entries from *Side_effects_unique*
- user_input_base = {'age': 61} : This should be changed to the subject's age
- <u>female</u>_col = next((c for c in sex_cols if <u>'female' in c.lower() and 'male'</u> not in c.lower()), None) if <u>female_col</u>: user_input_base[<u>female_col</u>] = 1 : Change all female lines to male and vice versa to input a male subject
- if '<u>SEMAGLUTIDE</u>' in col.upper(): user_input_base[col] = 1 : Change to the GLP-1 tested for

  

### Using Processed Data Running Models using Processed Data, and Model Results
Once data has successfully been pre-processed into a faers_with_embeddings_ready.csv file by running main.py, run the model_test.py in the testing directory to perform the training and vizualisation code on the pre-processed data.
  


<img width="1370" height="945" alt="image" src="https://github.com/user-attachments/assets/c0ac26b5-7e28-4c1d-9133-509a4b25845c" />


### Model Evolution
Originally, a **Random-Forest** model was used for analysis, however due to its poor results stemming from Random-Forest's aggressive guessing of "Yes" to satisfy the balance requirement, it led to many false positives. This was substituted with **XGBoost**, an algorithm  that uses an ensemble of decision trees to solve problems like classification and regression. It is known for its high performance, speed, and scalability, and can handle large datasets by using parallel and distributed computing [8]. 

However, after implementing this more robust model, I discovered that using MultiOutputClassifier in XGBoost creates a Binary Relevance problem. It trains a separate model for every single drug of which this list has several hundred. This led to an issue of training occuring for an unending length, obviously not realistic or achievable with this project's resources.

A **neural network** model was eventually chosen as a middle ground, as neural networks perform well on data with high-dimensional input and output space [9].

<img width="1527" height="972" alt="image" src="https://github.com/user-attachments/assets/1ed0d9ee-1a97-4fd9-8551-60c2cb23c11c" />

These results are much more consistent, although still lacking in very strong connections that could be marketable.

##Use the setting above within 'rf_model_test.py' to set testing parameters. This should be possible to implement on a webpage or application front-end in future iterrations.

### Video Explanation and Slides
Linked here are supplmental components for my course SAT 5141.
- Video description of the project's goal and history, conclusions, and very brief overview of functions. https://huskycast.hosted.panopto.com/Panopto/Pages/Viewer.aspx?id=a520f703-30a9-41b8-bc17-b3a90173469e
- Link to Google Drive slideshow paired with the video: https://docs.google.com/presentation/d/10FM04NeMmONqG19YPC1-_vg9YWEH-ABAQM5i9SyaZOk/edit?usp=sharing

### Research and Justifications

#### Dataset

The HODDI dataset is a decade of collected drug-report logs that contain 109,744 records involving 2,506 unique drugs and 4,569 unique side effects, specifically curated to capture multi-drug interactions and their collective impact on adverse effects [10]. High order drug interactions are drug combinations involving three or more drugs which produce an effect different from the sum of the individual drug effects. The main source of information for HODDI comes from the FAERS database from the U.S. Food and Drug Administration. The data structure is contained in CSV files with five essential columns detailing information about drug interaction reports: report identifier, recommended CUI, DrugBank ID, hyperedge labels (1 for positive samples, -1 for negative samples), and temporal information (year and quarter). This particular dataset is built on the back of previous datasets and studies including SIDER, OFFSIDES, TWOSIDES, and HPRD. The things this dataset is achieving to fix include lack of interaction lists, as separate datasets fail to wholly and completely capture the broad range of medications present in the market and all of their interactions and individual side effects. Additionally,  Inconsistent data quality across different sources (Ismail & Akram, 2022), affect the reliability of predictions with previous attempts, and finally Data sparsity issues in polypharmacy settings, where the number of possible drug combinations grows exponentially while available data remains limited (Tekin et al., 2017; 2018). They have benchmarked the HODDI data with three different model architectures: Multi-Layer Perceptron (MLP), Traditional Graph Neural Networks (GCN, GAT), and Hypergraph Architectures. Their best results came from using the HGNN-SA model, with 0.906 precision overall, F1 0.933, AUC 0.957, and PRAUC of 0.939. The researchers state in relation to the results, “The strong performance of both GNN models and the MLP model suggests that HODDI provides rich and high-quality feature representations, making it a valuable resource for advancing machine learning approaches in drug-drug interaction studies.”  HODDI can work as a basis for my analysis of adverse drug-drug interactions, providing others with the means to better predict the reasons behind their symptoms and if a change in medication can remediate them.

#### Target Population 
The population selected for this model comprises individuals diagnosed with Type 2 Diabetes Mellitus, Obesity (BMI ≥ 30 kg/m²), or any individuals who are currently prescribed Glucagon-like Peptide-1 Receptor Agonists (GLP-1 RAs) [1].  I chose this subject because the area of interest in GLP-1 drugs is exploding. According to Fang Heglanda, data is showing a doubling in use from 2019 to 2022, and more recently [11]. Usage data in 2024 showing that 1,449,442 patients were prescribed a GLP-1 RA between January 2018 and September 2024, with 6,341,367 total prescriptions during this period [12]. I have family members who take GLP-1 medications and the proper usage of this drug is key in continuing the fight against rising diabetes and obesity globally. 

#### High Risk Population
Cardiovascular disease is very righ risk in this population and GLP-1 agonists are a newer form of preventative measure against this for issues such as glycemic control [1]. The potential for high-order adverse drug reactions is elevated in this group, due to the newness of mass adoption of these drugs. Additionally, the direct medical costs and prevalence of cardiovascular conditions are projected to rise significantly through to 2050 [7]. Therefore, seeking to predict and prevent adverse drug reactions for this population is the focus of this application. Drug-related morbidity currently costs the U.S. healthcare system an estimated $528 billion annually [6].

#### Compound Formulation Risks
A distinct subset of this population takes  compounded semaglutide due to supply shortages and costs. Unlike FDA-approved products, these compounded formulations lack rigorous oversight and have been documented to contain incorrect salt forms (e.g., semaglutide sodium) or varying concentrations [5]. However, FAERS database  has been validated as a reliable source for detecting adverse drug reactions in GLP-1 users [2] is a baseline for detecting and possibly responding in emergent drug interaction events. This model is built to predict possible interactions from the use of specific GLP-1 agonists and their compounded forms in order to provide an on-site analysis using up to date medical data.

### Conclusions
The overall performance of the model is somewhat lacking in terms of precision and reliability. I believe this may be due to several factors. First, the idea of the application is difficult to implement, due to the vast quantity of drug-drug adverse reactions and how many of them are likely to be occuring with barebone inputs (nausea, dizziness, etc.). Due to this, a model's confidence in its predictions will almost always be incredibly low as it contains too many false negatives or incredibly high with false positive correlations. Narrowing this scope is incredibly difficult with my chosen parameters.

Another possibility is my lack of understanding in fully understanding model tuning. While this took many iterrations and changes to implement, I admit my newness to this field and study, and still found the model's performance on such a broad data set to be a good first step.

### References
[1] Lincoff, A. M., et al. (2023). Semaglutide and Cardiovascular Outcomes in Obesity without Diabetes. The New England Journal of Medicine, 389, 2221–2232. https://doi.org/10.1056/NEJMoa2307563
Relevance: Establishes the high-risk cardiovascular profile of the target population and the drug's efficacy, justifying the "High Risk" classification.

[2] Shu, Y., et al. (2022). Gastrointestinal adverse events associated with semaglutide: A pharmacovigilance study based on FDA Adverse Event Reporting System. Frontiers in Endocrinology, 13, 996179. https://doi.org/10.3389/fendo.2022.996179
Relevance: Primary literature analyzing the exact database (FAERS) you are using; validates the prevalence of GI adverse events.

[3] Hooper, A. J., & Liu, X. (2024). GLP-1RA-induced delays in gastrointestinal motility: Predicted effects on coadministered drug absorption by PBPK analysis. Clinical Pharmacology & Therapeutics. https://doi.org/10.1002/cpt.3188
Relevance: Provides the pharmacokinetic mechanism (delayed gastric emptying) that causes the drug-drug interactions your model seeks to predict.

[4] Sodhi, M., et al. (2023). Risk of Gastrointestinal Adverse Events Associated With Glucagon-Like Peptide-1 Receptor Agonists for Weight Loss. JAMA, 330(18), 1795–1797. https://doi.org/10.1001/jama.2023.19574
Relevance: Quantifies the risk of severe side effects (pancreatitis, gastroparesis) in the weight-loss population specifically.

[5] Spitery, A., et al. (2024). Legal, safety, and practical considerations of compounded injectable semaglutide. Journal of the American College of Clinical Pharmacy, 7(9), 941-946. https://doi.org/10.1002/jac5.1999
Relevance: Directly addresses the safety gap in compounded medications, a key differentiator of your project.

[6] Watanabe, J. H., McInnis, T., & Hirsch, J. D. (2018). Cost of Prescription Drug-Related Morbidity and Mortality. Annals of Pharmacotherapy, 52(9), 829–837. https://doi.org/10.1177/1060028018765159
Relevance: Provides the economic data on the cost of non-optimized drug therapy and adverse events.

[7] Martin, S. S., Aday, A. W., Almarzooq, Z. I., et al. (2024). 2024 Heart Disease and Stroke Statistics: A Report of US and Global Data From the American Heart Association. Circulation, 149. https://doi.org/10.1161/CIR.0000000000001209
Relevance: Provides the statistics for the projected economic burden of cardiovascular disease in the target population

[8] XGBoost explanation: https://www.geeksforgeeks.org/machine-learning/xgboost/

[9] Nam, J., Kim, J., Mencía, E. L., Gurevych, I., & Fürnkranz, J. (2014). Large-scale Multi-label Text Classification - Revisiting Neural Networks. Machine Learning and Knowledge Discovery in Databases, 437–452

[10] HODDI Dataset: https://github.com/TIML-Group/HODDI

[11] Hegland TA, Fang Z, Bucher K. GLP-1 Medication Use for Type 2 Diabetes Has Soared. JAMA. 2024;332(12):952–953. doi:10.1001/jama.2024.18219

[12] Samuel Gratzl, Patricia J Rodriguez, Brianna M Goodwin Cartwright, Charlotte Baker, Nicholas L Stucky medRxiv 2024.01.18.24301500; doi: https://doi.org/10.1101/2024.01.18.24301500 Mattingly TJ, Conti RM. Marketing and Safety Concerns for Compounded GLP-1 Receptor Agonists. JAMA Health Forum. 2025;6(1):e245015. doi:10.1001/jamahealthforum.2024.5015




