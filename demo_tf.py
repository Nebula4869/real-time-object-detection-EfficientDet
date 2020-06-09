import tensorflow as tf
import time
import cv2


def realtime_detection(camera_id, model_file_path):

    with tf.Session() as sess:
        with tf.gfile.FastGFile(model_file_path, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            tf.import_graph_def(graph_def, name='')

        graph = tf.get_default_graph()
        inputs = graph.get_tensor_by_name('image_arrays:0')
        predictions = graph.get_tensor_by_name('detections:0')

        classnames = ['background']
        with open('coco.names') as f:
            for name in f:
                classnames.append(name.split('\n')[0])

        cap = cv2.VideoCapture(camera_id)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        while cap.isOpened():
            _, frame = cap.read()

            start = time.time()

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            out = sess.run([predictions], feed_dict={inputs: [rgb_frame]})[0]

            for i in range(len(out)):

                left = int(out[0, i, 2])
                top = int(out[0, i, 1])
                right = int(out[0, i, 2] + out[0, i, 4] / 2)
                bottom = int(out[0, i, 1] + out[0, i, 3] / 2)

                score = out[0, i, 5]
                classname = classnames[int(out[0, i, 6])]

                cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
                cv2.putText(frame, '{}: {:.2f}'.format(classname, score), (left, top), cv2.FONT_HERSHEY_DUPLEX, (right - left) / 250, (0, 255, 0), 1)

            fps = 1 / (time.time() - start)
            cv2.putText(frame, 'FPS: {:.2f}'.format(fps), (0, 30), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 255, 0), 1)

            cv2.imshow('', frame)
            if cv2.waitKey(1) == 27:
                break

        cap.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    realtime_detection(1, './models/efficientdet-d0_frozen.pb')
