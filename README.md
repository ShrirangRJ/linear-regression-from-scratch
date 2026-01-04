A minimal linear regression implementation from scratch using NumPy to learn gradient descent concepts.

# Linear Regression From Scratch (Gradient Descent)

This project implements **linear regression from scratch using NumPy** — without using
any machine-learning libraries.

The goal was to understand:

- how gradient descent actually updates model weights  
- what “loss” means and how we minimize it  
- why models can explode (become `inf` / `nan`) if the learning rate is too high  

This is one of my first ML fundamentals projects while preparing for
LLM / AI research roles.

---

## 🧠 Problem

We generate simple synthetic data:
y = 3x + 2 + noise

Our model tries to learn:

y_pred = w * x + b


We train it using **Mean Squared Error (MSE)** and manually coded gradients.

 Implementation

Key steps implemented:

1️⃣ Create dataset  
2️⃣ Initialize weights (`w`) and bias (`b`)  
3️⃣ Compute predictions  
4️⃣ Calculate loss  
5️⃣ Compute gradients  
6️⃣ Update weights with gradient descent  

Everything is done manually so the learning mechanics are visible.

 Learning Rate Lesson

At first the model **diverged**:



loss → inf → nan
weights → nan


Reason: learning rate was too large.

After reducing it, training stabilized and the model learned values close to:



w ≈ 3
b ≈ 2


This same principle scales up to neural networks and LLMs.

---

 How to Run

### Install requirements

pip install numpy

Run the script
python linear_regression.py


You will see loss steadily decreasing and parameters converging.

📌 Output Example
step 0    | loss = 12.5
step 500  | loss ≈ small
...
w ≈ 3.0
b ≈ 2.0

