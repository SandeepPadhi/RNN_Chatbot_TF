# Building The Best ChatBot with Deep NLP



# Importing the libraries
from flask import Flask

import seq2seq_wrapper
import importlib
importlib.reload(seq2seq_wrapper)
import data_preprocessing
import data_utils_1
import data_utils_2
import tensorflow as tf
import numpy as np

#outputfinal=tf.placeholder()

########## PART 1 - DATA PREPROCESSING ##########



outfinal = tf.Variable(name="outfinal", shape=[20,1], dtype=tf.int32)
#tf.global_variables_initializer()


#outfinal=tf.Variable(tf.zeros([20,1],dtype=tf.int32),dtype=tf.int32,name="outfinal")


# Importing the dataset
metadata, idx_q, idx_a = data_preprocessing.load_data(PATH = './')

# Splitting the dataset into the Training set and the Test set
(trainX, trainY), (testX, testY), (validX, validY) = data_utils_1.split_dataset(idx_q, idx_a)

# Embedding
xseq_len = trainX.shape[-1]
yseq_len = trainY.shape[-1]
batch_size = 16
vocab_twit = metadata['idx2w']
xvocab_size = len(metadata['idx2w'])  
yvocab_size = xvocab_size
emb_dim = 1024
idx2w, w2idx, limit = data_utils_2.get_metadata()



########## PART 2 - BUILDING THE SEQ2SEQ MODEL ##########



# Building the seq2seq model
model = seq2seq_wrapper.Seq2Seq(xseq_len = xseq_len,
                                yseq_len = yseq_len,
                                xvocab_size = xvocab_size,
                                yvocab_size = yvocab_size,
                                ckpt_path = './weights',
                                emb_dim = emb_dim,
                                num_layers = 3)



########## PART 3 - TRAINING THE SEQ2SEQ MODEL ##########



# See the Training in seq2seq_wrapper.py



########## PART 4 - TESTING THE SEQ2SEQ MODEL ##########



# Loading the weights and Running the session
session = model.restore_last_session()

# Getting the ChatBot predicted answer


def respond(question):
    global session
    encoded_question = data_utils_2.encode(question, w2idx, limit['maxq'])
    print("Length of encoded_question is %d "%(len(encoded_question)))
    print("%s"%encoded_question)
    answer = model.predict(session, encoded_question)[0]
    #print(answer)
    answer=np.array(answer)
    print( answer)
    print ("Length of answer is %s"%(len(answer)))

   
    answer.reshape([20,1])
    print ("Shape of answer is %s"%answer.shape)
    
    outfinal.assign(answer)
    session.run(outfinal)
    print( outfinal)

    for i,j in enumerate(answer):
         print( "%d,%d"%(i,j))
         
         
    
    #session.run(outfinal)

    #a=outfinal.eval()
  
    #print( a)
    return data_utils_2.decode(answer, idx2w) 

# Setting up the chat
while True :
  question = input("You: ")
  answer = respond(question)
  print ("ChatBot: "+answer)






MODEL_NAME = 'Rajbot'
input_graph_path = 'model_files/savegraph.pbtxt'
checkpoint_path = 'model_files/model.ckpt'
input_saver_def_path = ""
input_binary = False
output_node_names = "output"
restore_op_name = "save/restore_all"
filename_tensor_name = "save/Const:0"
output_frozen_graph_name = 'model_files/frozen_model_'+MODEL_NAME+'.pb'
output_optimized_graph_name = 'model_files/optimized_inference_model_'+MODEL_NAME+'.pb'
clear_devices = True



freeze_graph.freeze_graph(input_graph_path, input_saver_def_path,
                          input_binary, checkpoint_path, output_node_names,
                          restore_op_name, filename_tensor_name,
                          output_frozen_graph_name, clear_devices, "")



output_graph_def = optimize_for_inference_lib.optimize_for_inference(
        sess.graph_def,
        ['en_{}'.format(t) for t in range(20)], # an array of the input node(s)
        ["output"], # an array of output nodes
        tf.float32.as_datatype_enum)
