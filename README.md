In order to recreate our work first download the fiddle dataset from this link: https://www.uio.no/ritmo/english/projects/mirage/databases/hf1/index.html. Unzip and configure your path to the file.
 Run the pre-processing.py with your path to the directory of the dataset. Below should be the correct files in a csv format.

'''
Train Data:
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
-----------------------------------------------------------------------------------
Test Data:
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
'''

Save those files in a directory names train_data and test_data which is located  

'''
C:pathtotheproject/Main/Data/
'''

Then run main_SMOTE.py to replicate our results.
