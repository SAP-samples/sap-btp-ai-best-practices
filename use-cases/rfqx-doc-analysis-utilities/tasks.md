Change Process Document Page:
- 1st: menu defining how many sources will there be
- 2st: for each source, create an upload box
  - The documents uploaded are not limited to PDFs but can also be excel/csv documents
    - Think how we could process the tabular files properly. A basic implementation would be to transform it into a pandas dataframe and then to text. 
  - All the documents uploaded for a given box should be related to the same provider
  - For this purpose, we might concatenate the strings of the different documents one another to have a single large document (GPT-4.1 has 1 Million token context window, should be able to handle it)
- These different documents should be used as a singular source of information for that provider
- Right in between the upload file boxes and the Process Document, there must be an expandable view of the features that will be extracted based on uc_rfq_4/rfq_schema.py. At the end of this list, there must be a text box allowing the user to add new features manually. Those features will be grouped as "Manually requested features" (just as the features in rfq_schema.py are already grouped in categories)
- **Feature extraction**: The application should not only extract the attributes in uc_rfq_4/rfq_schema.py but also analyze the features of said document that might not be part of the initial list. This should be grouped in "Dynamically fetched features". This capability should be by default OFF, and should be able to be turned on with a buttom check next to the Process Documents Buttoms.