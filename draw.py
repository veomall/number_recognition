import numpy as np
import matplotlib.pyplot as plt
from random import randint
from keras.datasets import mnist
from tensorflow import keras
from keras.layers import Dense, Flatten


(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normalize inputs
x_train = x_train / 255
x_test = x_test / 255

# Transform numbers to vectors of ones and zeros
# 7 -> [0, 0, 0, 0, 0, 0, 0, 1, 0, 0]
y_train_cat = keras.utils.to_categorical(y_train, 10)
y_test_cat = keras.utils.to_categorical(y_test, 10)

# Create model with 3 layers :
#     - inputs of 28*28 numbers
#     - hidden of 128 neurons
#     - outputs of 10 numbers
model = keras.Sequential([
    Flatten(input_shape=(28, 28, 1)),
    Dense(128, activation='relu'),
    Dense(10, activation='softmax')
])

print(model.summary())

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Training
model.fit(
    x_train,
    y_train_cat,
    batch_size=32,
    epochs=5,
    validation_split=0.2
)

# Testing
model.evaluate(x_test, y_test_cat)

# Predict random picture and compare prediction with right answer
n = randint(0, 10000)
x = np.expand_dims(x_test[n], axis=0)
res = model.predict(x)
print(res)

plt.imshow(x_test[n], cmap=plt.cm.binary)
plt.show()

print(np.argmax(res), '-------', y_test[n])

#%%
import tkinter as tk
import numpy as np

# Create a 28x28 array to store the pixel values
pixels = np.zeros((28, 28), dtype=np.uint8)

def draw_pixel(event):
    # Calculate the cell coordinates based on the mouse position
    cell_x = event.x // 10
    cell_y = event.y // 10

    # Update the pixel value in the array
    pixels[cell_y, cell_x] = 255

    # Draw the pixel on the canvas
    canvas.create_rectangle(cell_x * 10, cell_y * 10, (cell_x + 1) * 10, (cell_y + 1) * 10, fill='white')

def convert_to_numpy_array():
    # Convert the pixel values to a numpy array
    np_array = np.copy(pixels)

    # Display the numpy array
    print(np_array)

    x = np.expand_dims(np_array, axis=0)
    res = model.predict(x)
    print(res)

    plt.imshow(np_array, cmap=plt.cm.binary)
    plt.show()

    print(np.argmax(res))

def clear_canvas():
    # Clear the canvas
    canvas.delete("all")

    # Reset the pixel values to black
    pixels.fill(0)

# Create the main window
window = tk.Tk()
window.title("Pixel Drawing")
window.geometry("300x350")

# Create a canvas to draw on
canvas = tk.Canvas(window, width=280, height=280, bg='black')
canvas.pack()

# Bind the mouse motion event to the draw_pixel function
canvas.bind("<B1-Motion>", draw_pixel)

# Create a button to convert pixels to numpy array
convert_button = tk.Button(window, text="Convert to Numpy Array", command=convert_to_numpy_array)
convert_button.pack()

# Create a button to clear the canvas
clear_button = tk.Button(window, text="Clear Canvas", command=clear_canvas)
clear_button.pack()

# Start the main event loop
window.mainloop()
