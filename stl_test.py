import streamlit as st
from glob import glob
import numpy as np
import os
    
labels = ['HF', 'LF_ln', 'LF_noln', 'drone', 'car', 'animals', 'wind']
how_many_files = [1, 0, 2, 2, 10, 3, 5, 0]

all_files = sorted(glob(os.path.dirname(os.path.realpath(__file__))+'/**/*.wav', recursive = True))


def find_label(filename):
    return filename.split('/')[-2]


all_files = {label:[filename for filename in all_files if find_label(filename)==label] for label in labels}

def create_dataset():
    def create_test_dataset():
        dataset = []
        set_labels = []
        for nfiles, label in zip(how_many_files, labels):
            if nfiles>0:
                smallset = np.copy(all_files[label])
                np.random.shuffle(smallset)
                smallset = smallset[:nfiles]
                dataset.extend(smallset)
                set_labels.extend([label for i in smallset])
        shuffled_indices = np.arange(len(dataset))
        np.random.shuffle(shuffled_indices)
        dataset = np.array(dataset)[shuffled_indices]
        set_labels = np.array(set_labels)[shuffled_indices]
        return dataset, set_labels
    dataset, set_labels = create_test_dataset()
    responses = ['$---$' for label in set_labels]

    current_fileindex = 0
    current_filename = dataset[current_fileindex]
    return dataset, set_labels, responses, current_fileindex, current_filename




def app():
    responses_column = []
    res_cols = [st.sidebar.columns(3) for i in range(len(st.session_state['dataset']))]
    for icol in range(len(st.session_state['dataset'])):
        with res_cols[icol][1]:
            responses_column.append(st.empty())
            with responses_column[icol]:
                if icol == st.session_state['current_fileindex']:
                    st.markdown(f":red[{st.session_state['responses'][st.session_state['current_fileindex']]}]")
                else:
                    st.markdown(st.session_state['responses'][icol])
    for icol in range(len(st.session_state['dataset'])):
        with res_cols[icol][0]:
            if st.button(str(icol+1), use_container_width = True):
                with responses_column[st.session_state['current_fileindex']]:
                    st.markdown(st.session_state['responses'][st.session_state['current_fileindex']])
                st.session_state['current_fileindex'] = icol
                st.session_state['current_filename'] = st.session_state['dataset'][st.session_state['current_fileindex']]
                with responses_column[st.session_state['current_fileindex']]:
                    st.markdown(f":red[{st.session_state['responses'][st.session_state['current_fileindex']]}]")
                
        

    fileplace = st.empty()
    with fileplace:
        st.markdown(f"{st.session_state['current_fileindex']+1} / {len(st.session_state['dataset'])} files")
        
        
    st.audio(st.session_state['current_filename'])


    cols = []
    for i in range(len(labels)//3):
        these_cols = st.columns(3)
        cols.append(these_cols)

    if len(labels)%3!=0:
        cols.append(st.columns(len(labels)%3))

    for il, label in enumerate(labels):
        row, col = il//3, il%3
        with cols[row][col]:
            if st.button(label,  use_container_width = True):
                st.session_state['responses'][st.session_state['current_fileindex']] = label
                with responses_column[st.session_state['current_fileindex']]:
                    st.markdown(st.session_state['responses'][st.session_state['current_fileindex']])
                st.session_state['current_fileindex'] += 1
                if st.session_state['current_fileindex'] == len(st.session_state['dataset']):
                    st.session_state['current_fileindex'] = 0
                with responses_column[st.session_state['current_fileindex']]:
                    st.markdown(f":red[{st.session_state['responses'][st.session_state['current_fileindex']]}]")
                st.session_state['current_filename'] = st.session_state['dataset'][st.session_state['current_fileindex']]
                with fileplace:
                    st.markdown(f"{st.session_state['current_fileindex']+1} / {len(st.session_state['dataset'])+1} files")
                    
    if '$---$' not in st.session_state['responses']:
        st.empty()
        if st.button('done!',use_container_width = True):
            with responses_column[st.session_state['current_fileindex']]:
                st.markdown(f"{st.session_state['responses'][st.session_state['current_fileindex']]}")
                
            st.markdown('Results')
            for icol in range(len(st.session_state['dataset'])):
                with res_cols[icol][2]:
                    if st.session_state['set_labels'][icol] == st.session_state['responses'][icol]:
                        st.markdown(f":green[{st.session_state['set_labels'][icol]}]")
                    else:
                        st.markdown(f":red[{st.session_state['set_labels'][icol]}]")
            st.write('Correct responses per category:')
            for label in labels:
                st.write(f"{label}: {np.sum((np.array(st.session_state['responses']) == label) & (np.array(st.session_state['set_labels']) == label))} / {np.sum(np.array(st.session_state['set_labels']) == label)}")
                
                        
                    
                
                
if 'dataset' not in st.session_state:
    dataset, set_labels, responses, current_fileindex, current_filename = create_dataset()
    st.session_state['dataset'] = dataset
    st.session_state['set_labels'] = set_labels
    st.session_state['responses'] = responses
    st.session_state['current_fileindex'] = current_fileindex
    st.session_state['current_filename'] = current_filename
app()