import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import sqlite3 as lite
from datetime import datetime, timedelta, date

st.write("testing?")

def cmpltTask(task):
    idx = st.session_state.mytsks.index(task)
    st.session_state.chkarr[idx] = not st.session_state.chkarr[idx]
    st.session_state.rerun = True

def listTasks():
    st.session_state.tskclk = []
    for i, task in enumerate(st.session_state.mytsks):
        st.session_state.tskclk.append(st.checkbox(task, value = st.session_state.chkarr[i], key = 'l' + f'{dt.now():%d%m%Y%H%M%S%f}', on_change = cmpltTask, args=(task,), help ='Check to mark as completed'))


def main():
    st.title("Customer Database App")

    if "mytsks" not in st.session_state:
        st.session_state.mytsks = []

    if "tskclk" not in st.session_state:
        st.session_state.tskclk = []

    if "chkarr" not in st.session_state:
        st.session_state.chkarr = []

    if "rerun" not in st.session_state:
        st.session_state.rerun = False



    if st.session_state.rerun == True:
        st.session_state.rerun = False
        st.experimental_rerun()

    else:
        tsk = st.text_input('Enter your tasks', value="", placeholder = 'Enter a task')
        if st.button('Add Task'):
            if tsk != "":
                st.session_state.mytsks.append(tsk)
                st.session_state.chkarr.append(False)

    


if __name__ == '__main__':
    main()
    listTasks()