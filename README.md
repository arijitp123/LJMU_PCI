The Perspective Concentration Index (PCI) has been developed using the Informfully framework (https://github.com/Informfully).

To evaluate the PCI, we have used 2 baseline model viz. D-RDW and NRMS. 

Dataset used for evaluation is : MIND Small

The source has been tested using VS Code on Windows 11 OS with following configuration:
  Processor	13th Gen Intel(R) Core(TM) i5-1340P (1.90 GHz)
  Installed RAM	16.0 GB 
  System type	64-bit operating system, x64-based processor

Follow the steps given below :
1) Run this file to perform pre-processing of MIND dataset for D-RDW model compatibility: mind_to_drdw_preprocessor.py
2) Run this file to perform pre-processing of MIND dataset for NRMS model compatibility: mind_to_nrms_preprocessor.py
3) Run this file to perform ablation study of D-RDW and D-RDW+PCI (single -session): drdw_example_mind.py
4) Run this file to perform ablation study of D-RDW and D-RDW+PCI (multi-session): example_multi_session_pci.py
5) Run this file to perform ablation study of NRMS and NRMS+PCI (single -session): example_nrms_news_reranking.py
6) Run this file to perform ablation study of NRMS and NRMS+PCI (multi-session): example_nrms_multi_session_pci.py
