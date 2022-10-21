from extract_features import bertify
import tensorflow as tf

def run(model , config, json_object, cluster_key):
    
    with tf.Session() as session:
        writer = tf.summary.FileWriter("./log" , session.graph)
        model.restore(session)
        example = json_object
        tensorized_example = model.tensorize_example(example, is_training=False)
        if tensorized_example is None:
            example[cluster_key] = []
        else:
            
            feed_dict = {i:t for i,t in zip(model.input_tensors, tensorized_example)}
            _, _, _, top_span_starts, top_span_ends, top_antecedents, top_antecedent_scores = session.run(model.predictions, feed_dict=feed_dict)
            predicted_antecedents = model.get_predicted_antecedents(top_antecedents, top_antecedent_scores)
            example[cluster_key], _ = model.get_predicted_clusters(top_span_starts, top_span_ends, predicted_antecedents)
        
        if cluster_key == "predicted_clusters" and "clusters" in example:
             del example["clusters"]          
    
    return example 

def predict(json_object , model , config):
    
    must_bertify , cluster_key , config['lm_path'] = True , "clusters" , "bert_features_predict.hdf5"
    
    if must_bertify:
        model.bert_embedding_sentences = bertify(json_object)
          
    return run(model= model , config=config , json_object=json_object, cluster_key=cluster_key)