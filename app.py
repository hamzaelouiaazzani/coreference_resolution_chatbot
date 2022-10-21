import tensorflow as tf

from deployment import return_coreferenced_sentence , tokenize_text , make_json
from predict import predict
import util
from coref_model import CorefModel as cm


coref_model = "fr_mentcoref"
config = util.initialize_from_env(coref_model)
model = cm(config)







from flask import Flask, render_template, request, url_for, jsonify 
app = Flask(__name__)




coref_model = "fr_mentcoref"
config = util.initialize_from_env(coref_model)
model = cm(config) 
    
@app.route('/tests/endpoint', methods=['POST'])
def my_test_endpoint():
    
    
    
    input_props = []
    input_props.append((tf.string, [None, None]))  # Tokens.
    input_props.append((tf.float32, [None, None, model.context_embeddings.size]))  # Context embeddings.
    input_props.append((tf.float32, [None, None, model.head_embeddings.size]))  # Head embeddings.
    input_props.append((tf.float32, [None, None, model.lm_size, model.lm_layers]))  # LM embeddings.
    input_props.append((tf.int32, [None, None, None]))  # Character indices.
    input_props.append((tf.int32, [None]))  # Text lengths.
    input_props.append((tf.int32, [None]))  # Speaker IDs.
    input_props.append((tf.int32, []))  # Genre.
    input_props.append((tf.bool, []))  # Is training.
    input_props.append((tf.int32, [None]))  # Gold starts.
    input_props.append((tf.int32, [None]))  # Gold ends.
    input_props.append((tf.int32, [None]))  # Cluster ids.

    model.queue_input_tensors = [tf.placeholder(dtype, shape) for dtype, shape in input_props]
    dtypes, shapes = zip(*input_props)
    queue = tf.PaddingFIFOQueue(capacity=10, dtypes=dtypes, shapes=shapes)
    model.enqueue_op = queue.enqueue(model.queue_input_tensors)
    model.input_tensors = queue.dequeue()

    model.predictions, model.loss = model.get_predictions_and_loss(*model.input_tensors)
    model.global_step = tf.Variable(0, name="global_step", trainable=False)
    model.reset_global_step = tf.assign(model.global_step, 0)
    model.learning_rate = tf.train.exponential_decay(model.config["learning_rate"], model.global_step,
                                                    model.config["decay_frequency"], model.config["decay_rate"],
                                                    staircase=True)
    model.trainable_variables = tf.trainable_variables()
    gradients = tf.gradients(model.loss, model.trainable_variables)
    gradients, _ = tf.clip_by_global_norm(gradients, model.config["max_gradient_norm"])
    optimizers = {
        "adam": tf.train.AdamOptimizer,
        "sgd": tf.train.GradientDescentOptimizer
    }
    optimizer = optimizers[model.config["optimizer"]](model.learning_rate)
    opt_op = optimizer.apply_gradients(zip(gradients, model.trainable_variables), global_step=model.global_step)
    # Create an ExponentialMovingAverage object
    model.ema = tf.train.ExponentialMovingAverage(decay=model.config["ema_decay"])
    with tf.control_dependencies([opt_op]):
        model.train_op = model.ema.apply(model.trainable_variables)
    model.gold_loss = tf.constant(0.)

    # Don't try to restore unused variables from the TF-Hub ELMo module.
    vars_to_restore = [v for v in tf.global_variables() if "module/" not in v.name]
    if eval_mode:
        vars_to_restore = {model.ema.average_name(v): v for v in tf.trainable_variables()}
    model.saver = tf.train.Saver(vars_to_restore)

    if not eval_mode:
        # Make backup variables
        with tf.variable_scope('BackupVariables'):
            model.backup_vars = [tf.get_variable(var.op.name, dtype=var.value().dtype, trainable=False,
                                                initializer=var.initialized_value())
                                for var in model.trainable_variables]

        model.is_training = model.input_tensors[8]
        model.switch_to_train_mode_op = tf.cond(model.is_training, lambda: tf.group(), model.to_training)
        model.switch_to_test_mode_op = tf.cond(model.is_training, model.to_testing, lambda: tf.group())
    
    
    
    input_json = request.json
    input_list_text = input_json["text"]
    
    sents, pos, pars = tokenize_text(input_list_text, lang = "fr")
    conver_2_json_object = make_jsonlines(sents, pos, pars, fpath = "file", genre = "ge")
        
    coreferenced_json_object = predict(conver_2_json_object , model , config)
    last_question_coreferenced = return_coreferenced_sentence(coreferenced_json_object , input_list_text)
    
    
    return jsonify(last_question_coreferenced)


    
if __name__ == '__main__':
    app.run(debug=True , port = 5000)

