Click [here](https://github.com/AbuTaher003/Deep-Learning-DL/blob/main/Codes/26_Regularization.ipynb) for codes
---

# Regularization 

---

`আমাদের মডেল যদি overfit করে তাহলে আমরা Regularization ব্যবহার করে সেইটা থেকে মুক্ত হতে পারি । `


# Why Neural Network Overfit?

`আমরা যখন একটা Neural Network এ একটা node রাখি তখন সেইটা আমাদের একটা line draw করে দেয় শুধু । আমরা Neural Network যদি Neural Network এ node বা hidden layer বাড়াতে থাকি তাহলে আমাদের সেই line তত  বেশি curve হতে থাকে । আর আমাদের training data এর minor point গুলোও capture করতে থাকে।  `

![Alt text](img/image-144.png)


` overfitting এর সমস্যা দুর করার জন্য আমরা node কমাতে পারি । node কমাতে হলে আমাদের সেই node weight বা  piority কে আমরা শূন্য এর কাছাকাছি নিয়ে যাবো । `


# How to solve overfitting problem:

![Alt text](img/image-145.png)


# How regularization work?

![Alt text](img/image-146.png)

`আমরা Neural Network অথবা ANN এ weight and bias এর মান বের করি cost function এর মান reduce করার মাধ্যমে ।  আর regularization এ আমরা cost function এর সাথে একটা penalty add করে দেয় । L2 Regularization এর ক্ষেত্রে আমরা উপরের equation টি penalty এর মান এর সমান হিসেবে বসায়। এখানে, লেমডা হচ্ছে, hyperparameter । লেমডা এর মান বাড়ালে আমরা overfitting থকে underfitting এর দিকে চলে যায় ।  `

`penalty এর কাজ হচ্ছে আমাদের Neural Network এর weight গুলোর মান কে 0 এর  দিকে নিয়ে আসে । `

` L1 এর ক্ষেত্রে শুধু  square টা হবে না ।  `

![Alt text](img/image-147.png)


` আবার অনেক জায়গায় আমরা সমীকরণটি নিচের মতো দেখতে পাবো । `

![Alt text](img/image-148.png)

`যেখানে, penalty এর equation এ l বলতে বুঝায় আমরা কোন layer এ আছি । তারপর, i আর j দিয়ে বুঝায়  Neural Network এর (i-তম column, j-তম row ) এর node । `



# Intuition Behind Regularization:

![Alt text](img/image-149.png)

`Regularization এ আমরা loss function এর সাথে একটা penalty add করতেছি । আর আমরা কোন এর ভ্যালু চেইজ করার জন্য আমরা আগের এর ভ্যালু এর সাথে learning rate * সেই weight এর সাপেক্ষে loss function(with regularization) এর differentiation । Finally, আমরা যেই ans পাচ্ছি সেইটা (1- learning rate * hyperparameter)*initial weight আর hyperparameter এর মান positive হয় । অনেক জায়গায় (1- learning rate * hyperparameter) কে weight decay বলে । অনেক সময়  L2 regularization কে weight decay বলে । `




