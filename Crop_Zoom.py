import tensorflow as tf


def mkMask(m, n, d):
    M = tf.range(start=0, limit=n, dtype=tf.float32)
    rt = tf.stack([M] * m)
    return tf.stack([rt] * d, axis=2)


def h(x):
    return 1 / (1 + tf.exp(-10 * x))


def zoom_in(tensor, locations):
    shapes = tensor.get_shape().as_list()
    batches = shapes[0]
    img_size = shapes[1]

    box_ids = tf.range(0, batches)
    locations = tf.cast(locations, tf.float32)
    boxes = locations / img_size
    return tf.image.crop_and_resize(tensor, boxes, box_ids, (img_size, img_size))


def Crop_forward(tensor, attention):
    shapes = tensor.get_shape().as_list()
    batches = shapes[0]
    in_size = shapes[1]

    x = mkMask(in_size, in_size, 3)
    y = mkMask(in_size, in_size, 3)

    rt = []

    for i in range(batches):
        tx, ty, tl = attention[i][0], attention[i][1], attention[i][2]
        tx = tf.cond(tf.greater(tx, in_size / 3), lambda: tx, lambda: tf.constant(in_size / 3))
        ty = tf.cond(tf.greater(ty, in_size / 3), lambda: ty, lambda: tf.constant(in_size / 3))
        tl = tf.cond(tf.greater(tl, in_size / 3), lambda: tl, lambda: tf.constant(in_size / 3))

        txtl = tf.cond(tf.greater(tx - tl, 0), lambda: tx - tl, lambda: tf.convert_to_tensor(0, dtype=tf.float32))
        tytl = tf.cond(tf.greater(ty - tl, 0), lambda: ty - tl, lambda: tf.convert_to_tensor(0, dtype=tf.float32))

        txbr = tf.cond(tf.less(tx + tl, in_size), lambda: tx + tl,
                       lambda: tf.convert_to_tensor(in_size, dtype=tf.float32))
        tybr = tf.cond(tf.less(ty + tl, in_size), lambda: ty + tl,
                       lambda: tf.convert_to_tensor(in_size, dtype=tf.float32))
        mk = (h(x - txtl) - h(x - txbr)) * (h(y - tytl) - h(y - tybr))

        xatt = tensor[i] * mk
        xamp = zoom_in(tf.stack([xatt]), tf.stack([tf.stack([txtl, tytl, txbr, tybr])]))

        xamp = tf.squeeze(xamp, 0)
        rt.append(xamp)

    return tf.convert_to_tensor(rt)


if __name__ == '__main__':


    # a = tf.convert_to_tensor(0)
    #
    # def body(i, n):
    #     return tf.add(i, 1), n
    #
    # ii, nn = tf.while_loop(lambda i, n: tf.less(i,n), body, [a, 1000000])
    #
    # with tf.Session() as sess:
    #     m = sess.run(ii)
    #     print(m)
    # a = [[1 ,2]]
    # b = [[2 , 3]]
    # c = tf.concat([a,b],0)
    # print(c)


    # a = tf.Variable(tf.zeros([3,10,10,3]))
    # b = tf.ones([1,10,10,3])
    #
    # c = tf.scatter_update(a, [0], b)
    #
    # sess = tf.Session()
    # sess.run(tf.global_variables_initializer())
    # cc = sess.run(c)
    # print(cc)

    from Read import main
    import numpy as np
    a = tf.compat.v1.placeholder(
        tf.compat.v1.float32,
        [2, 224, 224, 3])
    b = [[100, 100, 200], [30, 30, 20]]
    rt = Crop_forward(a, tf.convert_to_tensor(b, dtype=tf.float32))

    print(rt)
    import cv2
    with tf.compat.v1.Session() as sess:
        sess.run(tf.global_variables_initializer())
        tt = main(2, './train/')[0]
        cc = sess.run(rt, feed_dict={a: tt})

        print(type(cc))
        for i in range(len(cc)):
            cv2.imshow(str(i), np.uint8(cc[i] * 255.0))
            cv2.imshow(str(i) + 'main', np.uint8(tt[i] * 255.0))
        cv2.waitKey(0)