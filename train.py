import tensorflow as tf
import input
import FLAGS
import inference
import time as t
start = t.time()

def loss(real_images, latent, t):
    # WGAN-GP Paper Notation
    x = real_images
    x_tilda = generator_network(latent, TARGET_RES, t)
    epsilon = tf.random_uniform([BATCH_SIZE, 1, 1, 1], minval=0.0, maxval=1.0)
    x_hat = epsilon * x + (1 - epsilon) * x_tilda

    with tf.variable_scope('Scoring'):
        score_x_hat = discriminator_network(x_hat, t)
        score_x = discriminator_network(x, t)
        score_x_tilda = discriminator_network(x_tilda, t)

    tf.summary.image("GenImages", x_tilda)
    #tf.summary.image("RealImages", x)

    with tf.variable_scope("Loss"):
        grads = tf.gradients(score_x_hat, [x_hat])[0]
        gradient_penalty = 10.0*tf.square(tf.sqrt(tf.reduce_sum(tf.square(grads), axis=[1, 2, 3])) - 1)
        epsilon_cost = 0.001 * tf.square(score_x)
        d_loss = tf.reduce_mean(score_x_tilda-score_x+epsilon_cost+gradient_penalty)
        g_loss = tf.reduce_mean(-score_x_tilda)
        #d_loss = -tf.reduce_mean(tf.log(score_x) + tf.log(1. - score_x_tilda))
        #g_loss = -tf.reduce_mean(tf.log(score_x_tilda))
        

    tf.summary.scalar("D-Loss", d_loss)
    tf.summary.scalar("Score_Of_Real_Images", tf.reduce_mean(score_x))
    tf.summary.scalar("Score_Of_Generator_Images", tf.reduce_mean(score_x_tilda))
    tf.summary.scalar("Distance", tf.reduce_mean(score_x_tilda - score_x))
    tf.summary.scalar("GP", tf.reduce_mean(gradient_penalty))
    tf.summary.scalar("G-Loss", g_loss)
    tf.summary.scalar("Epsilon_Cost", tf.reduce_mean(epsilon_cost))

    return d_loss, g_loss
	

	def get_t(current_shown_images, dataset_size, epochs_for_layer_addition, epochs_for_fading):

    images_for_layer_addition = dataset_size * epochs_for_layer_addition
    images_for_fading = dataset_size * epochs_for_fading

    phase_duration = images_for_layer_addition + images_for_fading
    current_phase = int(current_shown_images/phase_duration)
    current_phase_percent = max(current_shown_images - current_phase*phase_duration - images_for_layer_addition, 0) / images_for_fading
    return current_phase + current_phase_percent

def train():

    print("Building graph ...")
    images = get_image_batch()
    latent = tf.random_uniform([BATCH_SIZE, 16, 16, 2])
    t_tensor = tf.placeholder(tf.float32, name='t_tensor', shape=[])
    tf.summary.scalar("t", t_tensor)
    d_loss, g_loss = loss(images, latent, t_tensor)
    
    d_trainable_vars = tf.trainable_variables(scope=".*Discriminator.*")
    g_trainable_vars = tf.trainable_variables(scope=".*Generator.*")
    
    d_optimizer = tf.train.AdamOptimizer(beta1 = 0.0)
    d_grads, d_vars = zip(*d_optimizer.compute_gradients(d_loss, var_list=d_trainable_vars))
    d_clipped_grads, _ = tf.clip_by_global_norm(d_grads, 5.0)
    [tf.summary.histogram('DGradient%d' % i, tf.reshape(d_clipped_grads[i], [-1])) for i in range(0, len(d_clipped_grads))]
    d_train_op = d_optimizer.apply_gradients(zip(d_clipped_grads, d_vars))
    
    g_optimizer = tf.train.AdamOptimizer(beta1 = 0.0)
    g_grads, g_vars = zip(*g_optimizer.compute_gradients(g_loss, var_list=g_trainable_vars))
    g_clipped_grads, _ = tf.clip_by_global_norm(g_grads, 5.0)
    [tf.summary.histogram('GGradient%d' % i, tf.reshape(g_clipped_grads[i], [-1])) for i in range(0, len(g_clipped_grads))]
    g_train_op = g_optimizer.apply_gradients(zip(g_clipped_grads, g_vars))
    
    merged_summary_op = tf.summary.merge_all()
    summary_counter = 0
    last_gtrain_run = 0
    
    print("Initiating session ...")
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        tf.train.start_queue_runners(sess=sess)

        total_images_shown = 13266368
        writer = tf.summary.FileWriter(SUMMARIES_DIR + '/train', sess.graph)
        saver = tf.train.Saver()
        saver.restore(sess, '../input/nonstop68/tmp/model.ckpt')
        print("Training ...")
        
        while (t.time() - start) < 5.8*60.0*60.0:
            t_py = get_t(total_images_shown, DATASET_SIZE, 3, 3)
            #t_py = 1000.0
            sess.run([d_train_op], feed_dict={t_tensor: t_py})
            total_images_shown += BATCH_SIZE
            
            if (total_images_shown - last_gtrain_run) > BATCH_SIZE*5:
                last_gtrain_run = total_images_shown
                sess.run([g_train_op], feed_dict={t_tensor: t_py})
                
            if summary_counter > 2000:
                summary_counter = 0
                print("Saving Summary ... ")
                sum = sess.run(merged_summary_op, feed_dict={t_tensor: t_py})
                writer.add_summary(sum, global_step=total_images_shown)
                print("Images trained on ", total_images_shown)
                
            
            summary_counter += BATCH_SIZE
            
        print("Finished training on", total_images_shown)
        save_path = saver.save(sess, "tmp/model.ckpt")
        print("Model saved in path: %s" % save_path)
        print("Finished")