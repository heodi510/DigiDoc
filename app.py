from typing import Dict
import streamlit as st
from PIL import Image
import sys
import pandas as pd
import base64

@st.cache(allow_output_mutation=True)
def get_static_store() -> Dict:
    """This dictionary is initialized once and can be used to store the files uploaded"""
    return {}

def main():
    """Run this function to run the app"""
    
    # SIDEBARS
    st.sidebar.header("Navigation")
    st.sidebar.markdown("We help convert menus, documents, invoices or other physical text into an organized CSV file.")
    document_type = st.sidebar.selectbox("Document Type",["Menus","Invoices (Coming Soon)","Tax Forms (Coming Soon)","Contracts (Coming Soon)","Reports (Coming Soon)","Id Documents (Coming Soon)"])
    # if st.sidebar.checkbox("Format Guide"):
    #     st.sidebar.checkbox("Format 1")
    st.sidebar.radio("Type of Format",("Format 1","Format 2","Format 3","Format 4"))
    st.sidebar.header("About")
    st.sidebar.info("***Input Your Own Description***")

    st.image("Document Digitization.jpg",width=700)
    st.title("Document Digitization")
    
    result = st.file_uploader("Upload one or more images to convert to CSV", type=["png","jpg","jpeg"],accept_multiple_files=True)

    if result:
        st.info("Total: " + str(len(result)) + " Images")
        st.info(result)

    if st.checkbox("Show Images / Hide Images"):
        for x in result:
            st.image(x.getvalue(),width=200)
    
    # ADD MODEL HERE
    if st.button("Run Model"):
        st.text('***LOAD MODEL HERE***')
        
        #PROGRESS BAR Take out if cannot figure out
        import time
        my_bar = st.progress(0)
        for p in range(100):
            time.sleep(0.5)
            my_bar.progress(p + 1)


    # CHANGE TO CSV FILE AND INPUT HERE
    data = [(1, 2, 3)]
    # When no file name is given, pandas returns the CSV as a string, nice.
    df = pd.DataFrame(data, columns=["Col1", "Col2", "Col3"])
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()  # some strings <-> bytes conversions necessary here
    #Change b64 to output file
    href = f'<a href="data:file/csv;base64,{b64}">Download CSV File</a> (right-click and save as &lt;some_name&gt;.csv)'
    st.markdown(href, unsafe_allow_html=True)

main()


# try:
#     import streamlit as st
#     import os
#     import sys
#     import pandas as pd
#     from io import BytesIO, StringIO 
#     print("All Modules Loaded ")
# except Exceptions as e:
#     print("Some Modules are Missing : {} ".format(e))

# STYLE = """
# <style>
# img {
#     max-width: 100%;
# }
# </style>
# """


# def main():
#     """Run this function to display the Streamlit app"""
#     st.info(__doc__)
#     st.markdown(STYLE, unsafe_allow_html=True)
#     file = st.file_uploader("Upload file", type=["csv","png","jpg"])
#     show_file = st.empty()

#     if not file:
#         show_file.info("Please Upload a file : {} ".format(' '.join(["csv","png","jpg"])))
#         return
#     content = file.getvalue()

#     # if isinstance(file, BytesIO):
#     #     show_file.image(file)
#     # else:
#     #     df = pd.read_csv(file)
#     #     st.dataframe(df.head(10))
#     file.close()
#     st.image(content)
#     st.text(path.content)

# main()


# # Text/Title
# st.title("Streamlit Tutorials")

# # Header/Subheader
# st.header("This is a header")
# st.subheader("This is a subheader")

# # Text
# st.text("Hello St")

# # Markdown
# st.markdown("### This is a Markdown")

# # Error/Colorful Text
# st.success("Successful")

# st.info("Information")

# st.warning("This is a warning")

# st.error("This is an error Danger")

# st.exception("NameError('name three not defined')")

# # Get Help Info About Python
# st.help(range)


# # Writing Text/Super Fxn
# st.write("text with write")

# st.write(range(10))

# # Images 
# st.image("images/5.jpg",width=300,caption="Menu Image")

# # Videos
# vid_file = open("sample.mp4","rb").read()
# st.video(vid_file)

# # Audio
# # audio_file = open("examplemusic.mp3","rb").read()
# # st.audio(audio_file,format='audio/mp3')


# # Widget
# # Checkbox
# if st.checkbox("Show/Hide"):
#     st.text("Showing or Hiding Widget")


# # Radio Buttons
# status = st.radio("What is your status",("Active","Inactive","Type 3","Type 4"))

# if status == 'Active':
#     st.success("You are Active")
# else:
#     st.warning("Not Active")

# # SelectBox
# occupation = st.selectbox("Your Occupation",["Programmer","Data Scientist","Doctor","Businessman"])
# st.write("You selected this option",occupation)

# # MultiSelect
# location = st.multiselect("Where do you work?",("Central","Taikoo","KwunTong","Causeway","Shatin"))
# st.write("You selected",len(location),'locations')

# # Slider

# level = st.slider("What is your level",1,5)

# # Buttons
# st.button("Simple Button")

# if st.button("About"):
#     st.text("Streamlit is Cool")

# # Text Input
# firstname = st.text_input("Enter Your First Name:")
# if st.button("Submit"):
#     result = firstname.title()
#     st.success(result)

# # Text Area
# message = st.text_area("Enter Your message","Type Here..")
# if st.button("Print"):
#     result = message.title()
#     st.success(result)

# # Date Input
# import datetime
# today = st.date_input("Today is",datetime.datetime.now())

# # Time
# the_time = st.time_input("The time is ",datetime.time())

# # Displaying JSON
# st.text("Display JSON")
# st.json({'name':"Kelvin",'gender':"male"})

# # Display Raw Code
# st.text("Display Raw Code")
# st.code("import numpy as np")

# # Display Raw Code
# with st.echo():
#     # This will also show
#     import pandas as pd
#     df = pd.DataFrame()

# # Progress Bar
# import time
# my_bar = st.progress(0)
# for p in range(10):
#     my_bar.progress(p + 1)

# # Spinner
# with st.spinner("Waiting .."):
#     time.sleep(5)
# st.success("Finished")

# # Balloons
# # st.balloons()

# # SIDEBARS
# st.sidebar.header("About")
# st.sidebar.text("This is Streamlit Tutorial")

# # Functions
# @st.cache
# def run_fxn():
#     return range(100)

# st.write(run_fxn())

# # Plot
# st.pyplot()

# # DataFrames
# st.dataframe(df)

# # Tables
# st.table(df)
