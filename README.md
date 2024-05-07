In order to recreate our work first download the fiddle dataset from this link: https://www.uio.no/ritmo/english/projects/mirage/databases/hf1/index.html. Unzip the file. Configure your path to the file
 and run the pre-processing.py with your configured path to the directory of the dataset in your computer. Below should be the correct files in a csv format.


Train Data:
```
Haslebuskane_angry_start_end_spect_target
Haslebuskane_happy_start_end_spect_target
Haslebuskane_original_start_end_spect_target
Haslebuskane_sad_start_end_spect_target
Haslebuskane_tender_start_end_spect_target
Havbrusen_angry_start_end_spect_target
Havbrusen_happy_start_end_spect_target
Havbrusen_original_start_end_spect_target
Havbrusen_sad_start_end_spect_target
Havbrusen_tender_start_end_spect_target
IvarJorde_angry_start_end_spect_target
IvarJorde_happy_start_end_spect_target
IvarJorde_original_start_end_spect_target
IvarJorde_sad_start_end_spect_target
IvarJorde_tender_start_end_spect_target
LattenSomBedOmNoko_angry_start_end_spect_target
LattenSomBedOmNoko_happy_start_end_spect_target
LattenSomBedOmNoko_original_start_end_spect_target
LattenSomBedOmNoko_sad_start_end_spect_target
LattenSomBedOmNoko_tender_start_end_spect_target
SigneUladalen_angry_start_end_spect_target
SigneUladalen_happy_start_end_spect_target
SigneUladalen_original_start_end_spect_target
SigneUladalen_sad_start_end_spect_target
SigneUladalen_tender_start_end_spect_target
```

Test Data:
```
Silkjegulen_angry_start_end_spect_target
Silkjegulen_happy_start_end_spect_target
Silkjegulen_original_start_end_spect_target
Silkjegulen_sad_start_end_spect_target
Silkjegulen_tender_start_end_spect_target
Valdresspringar_angry_start_end_spect_target
Valdresspringar_happy_start_end_spect_target
Valdresspringar_original_start_end_spect_target
Valdresspringar_sad_start_end_spect_target
Valdresspringar_tender_start_end_spect_target
Vossarull_angry_start_end_spect_target
Vossarull_happy_start_end_spect_target
Vossarull_original_start_end_spect_target
Vossarull_sad_start_end_spect_target
Vossarull_tender_start_end_spect_target
```

Save those files in a directory named train_data and test_data which is located  here:

```
C:/pathtotheproject/Main/Data/
```

Then, install the requirements.txt file. You can alternatively run the following commands in your terminal of your project:
```
pip install numpy
pip install jupyter
pip install librosa
pip install matplotlib
pip install scipy
pip install torch
pip install tensorboard-logger
pip install pandas
pip install tensorflow
pip install imbalanced-learn
```
Then run on Main/main_SMOTE.py to replicate our results. Although since our results are not good, feel free to try other hyperparameters as well.

In the Main folder exists our project with data exploration, pre-processing files and source code.

In the unused_code folder, exists unused code and an early project that is fully functional and is about instrument identification but was discontinued. It was about the original dataset on Kaggle: https://www.kaggle.com/datasets/imsparsh/musicnet-dataset?resource=download
You can find it in unused_code/Pytorch/Main.ipynb and run the file with configuration of the data as well. If you want to run it, you need to add a data directory inside the unused_code/Pytorch named Data, add the data from the kaggle dataset inside the directory and then run the jupyter notebook Main.ipynb. 

Thank you for reading!
