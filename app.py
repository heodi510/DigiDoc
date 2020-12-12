# pylint: disable=line-too-long
"""This is example shows how to **upload multiple files** via the
[File Uploader Widget](https://streamlit.io/docs/api.html?highlight=file%20upload#streamlit.file_uploader)

As far as I can see you can only upload one file at a time. So if you need multiple files in your
app, you need to store them in a static List or Dictionary. Alternatively they should be uploaded
as one .zip file.

Please note that file uploader is a **bit problematic** because
- You can only upload one file at the time.
- You get no additional information on the file like name, size, upload time, type etc. So you
cannot distinguish the files without reading the content.
- every time you interact with any widget, the script is rerun and you risk processing or storing
the file again!
- The file uploader widget is not cleared when you clear the cache and there is no way to clear the
file uploader widget programmatically.

Based on the above list I created [Issue 897](https://github.com/streamlit/streamlit/issues/897)

This example was based on
[Discuss 1445](https://discuss.streamlit.io/t/uploading-multiple-files-with-file-uploader/1445)
"""
# pylint: enable=line-too-long
from typing import Dict
import os
import io
import streamlit as st
from PIL import Image

@st.cache(allow_output_mutation=True)
def get_static_store() -> Dict:
    """This dictionary is initialized once and can be used to store the files uploaded"""
    return {}


def main():
    """Run this function to run the app"""
    static_store = get_static_store()

    st.info(__doc__)
    result = st.file_uploader("Upload", type=["png","jpg","jpeg"])
    if result:
        # Process you file here
        value = result.getvalue()

        # And add it to the static_store if not already in
        if not value in static_store.values():
            static_store[result] = value
    else:
        static_store.clear()  # Hack to clear list if the user clears the cache and reloads the page
        st.info("Upload one or more `.py` files.")

    if st.button("Clear file list"):
        static_store.clear()
    if st.checkbox("Show file list?", True):
        st.write(list(static_store.keys()))
    if st.checkbox("Show content of files?"):
        for value in static_store.values():
            st.code(value)
    
    if st.button("Show Image"):
        img = Image.open(result)
        st.info(len(static_store))
        for i,value in enumerate(static_store.values()):
            st.image(value,width=200)
            image = Image.open(io.BytesIO(value))
            image_name = 'image_'+str(i)+'.jpg'
            image.save('data/input_img/'+image_name)
            
            
    if st.button("run pipeline"):
        os.system('python CRAFT/pipeline.py')
        os.system('python CRAFT/crop_image.py')


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
