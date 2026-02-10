import os
# 引入tensorflow
from tensorflow.python.framework import ops
import tensorflow.compat.v1 as tf
tf.compat.v1.disable_eager_execution()

__EXPORT_ROOT_DIR_PATH__ = "/export/App/workspace/test/python/models"

asset_file1_path = "/export/App/workspace/test/python/data/master/lookup_dict_1.txt"
content_1 = '\n'.join(["emerson 1000", "lake 2000", "palmer 3000",])
with open(asset_file1_path, 'wb') as file:
    file.write(content_1.encode('utf-8'))

asset_file2_path = "/export/App/workspace/test/python/data/master/lookup_dict_2.txt"
content_2 = '\n'.join(["emerson 1000", "lake 2000", "palmer 3000",])
with open(asset_file2_path, 'wb') as file:
    file.write(content_2.encode('utf-8'))


###################### LookUp Table 操作 #####################
def LookUpTableOperation(asset_file_path, key_str):
    #tf.add_to_collection(tf.GraphKeys.ASSET_FILEPATHS, tf.constant([asset_file_path]))
    init = tf.lookup.TextFileInitializer(
        filename=asset_file_path,
        key_dtype=tf.string, key_index=0,
        value_dtype=tf.int32, value_index=1,
        delimiter=" ")
    
    table = tf.lookup.StaticHashTable(init, default_value=-1)
    table_val = table.lookup(tf.constant(key_str))
    
    return table_val
    

#################################################################################
########################### Saved Model: master model ###########################
#################################################################################
# Create a new graph
master_graph = tf.Graph()
# Use the graph context manager to define operations in the new graph
with master_graph.as_default():
    ########################### multiple tensor ###########################
    # 创建两个常量 Tensor
    const1 = tf.constant([[2, 2]], dtype=tf.int32)
    const2 = tf.constant([[4],
                          [4]], dtype=tf.int32)
    multiple = tf.matmul(const1, const2)
    
    # 尝试用print输出multiple的值
    print(multiple)

    table_val_1 = LookUpTableOperation(asset_file1_path, 'palmer')
    table_val_2 = LookUpTableOperation(asset_file2_path, 'emerson')
    table_val = tf.add(tf.reshape(table_val_1, []), tf.reshape(table_val_2, []))

    sum_ret = tf.add(tf.reshape(multiple, []), tf.reshape(table_val, []))


master_model_path = os.path.join(__EXPORT_ROOT_DIR_PATH__, "master_model")
with tf.Session(graph=master_graph) as sess:
    sess.run(tf.global_variables_initializer())
    sess.run(tf.tables_initializer())
    sess.run(sum_ret)
    
    builder = tf.saved_model.builder.SavedModelBuilder(master_model_path)
    #ops.add_to_collection("my_collection", tf.group(*tf.get_collection(tf.GraphKeys.TABLE_INITIALIZERS)))
    builder.add_meta_graph_and_variables(
        sess,
        [tf.saved_model.tag_constants.SERVING],
        signature_def_map={
            "serving_default": tf.saved_model.signature_def_utils.predict_signature_def(
                                   inputs={"input_1": const1, "input_2": const2}, 
                                   outputs={"output": sum_ret})},
        assets_collection=tf.get_collection(tf.GraphKeys.ASSET_FILEPATHS),
        legacy_init_op=tf.group(*tf.get_collection(tf.GraphKeys.TABLE_INITIALIZERS)))
    builder.save(as_text=True)
