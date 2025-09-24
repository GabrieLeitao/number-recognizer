import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf


train = False
num_epochs = 4 
image_size = 28


def preprocess_drawn(img):
    # Find bounding box
    x, y, w, h = cv.boundingRect(img)
    digit = img[y:y+h, x:x+w]

    # Preserve aspect ratio: scale max dimension to 20
    if w > h:
        new_w = 20
        new_h = int(h * (20 / w))
    else:
        new_h = 20
        new_w = int(w * (20 / h))
    digit = cv.resize(digit, (new_w, new_h), interpolation=cv.INTER_AREA)

    # Place into 28×28 canvas centered
    canvas = np.zeros((28, 28), dtype=np.uint8)
    x_offset = (28 - new_w) // 2
    y_offset = (28 - new_h) // 2
    canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = digit

    # Normalize
    canvas = canvas.astype('float32') / 255.0
    return np.expand_dims(canvas, axis=(0, -1))  # shape (1,28,28,1)



def draw_and_predict(model):
    drawing = False
    ix, iy = -1, -1
    canvas = np.zeros((280, 280), dtype=np.uint8)

    def draw(event, x, y, flags, param):
        nonlocal drawing, ix, iy, canvas
        if event == cv.EVENT_LBUTTONDOWN:
            drawing = True
            ix, iy = x, y
        elif event == cv.EVENT_MOUSEMOVE:
            if drawing:
                cv.line(canvas, (ix, iy), (x, y), 255, 20)
                ix, iy = x, y
        elif event == cv.EVENT_LBUTTONUP:
            drawing = False

    cv.namedWindow('Draw Digit')
    cv.setMouseCallback('Draw Digit', draw)

    while True:
        cv.imshow('Draw Digit', canvas)
        key = cv.waitKey(1) & 0xFF
        if key == ord('c'):
            canvas[:] = 0
        elif key == ord('p'):
            img = preprocess_drawn(canvas)
            prediction = model.predict(img)
            print(f'Predicted digit: {np.argmax(prediction)}')
            plt.imshow(img[0], cmap=plt.cm.binary)
            plt.title(f'Prediction: {np.argmax(prediction)}')
            plt.show()
        elif key == 27:  # ESC to exit
            break
    cv.destroyAllWindows()


if train:

    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # scale down the values to 0 to 1
    x_train = tf.keras.utils.normalize(x_train, axis=1)
    x_test = tf.keras.utils.normalize(x_test, axis=1)

    # CNN with 1.2M parameters
    model = tf.keras.Sequential([
        tf.keras.layers.Reshape((28, 28, 1), input_shape=(28, 28)),

        tf.keras.layers.Conv2D(32, (3,3), activation='relu', padding='same'),
        tf.keras.layers.MaxPooling2D((2,2)),
        tf.keras.layers.Dropout(0.25),

        tf.keras.layers.Conv2D(64, (3,3), activation='relu', padding='same'),
        tf.keras.layers.MaxPooling2D((2,2)),
        tf.keras.layers.Dropout(0.25),

        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu',
                            kernel_regularizer=tf.keras.regularizers.l2(1e-4)),
        tf.keras.layers.Dropout(0.5),

        tf.keras.layers.Dense(10, activation='softmax')
    ])

    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    model.fit(x_train, y_train, epochs=num_epochs)

    loss, accuracy = model.evaluate(x_test, y_test)
    print(accuracy)
    print(loss)

    model.save('digits.keras')

else:
    # Load previously saved model
    model = tf.keras.models.load_model('digits.keras')

if model == None:
    print("Could not load model")
else:
    # Test with some images
    for i in range(10):
        img = cv.imread(f'{i}.png')[:,:,0]
        img = np.invert(np.array([img]))
        prediction = model.predict(img)
        print(f' The result is probably: {np.argmax(prediction)}')
        # plt.imshow(img[0], cmap=plt.cm.binary)
        # plt.show()
    # Add real-time drawing and prediction
    draw_and_predict(model)
