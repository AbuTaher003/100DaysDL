# Detailed Notes on Types of Neural Networks

## ১. Feedforward Neural Network (FNN)

### সংজ্ঞা:

Feedforward Neural Network হলো সবচেয়ে মৌলিক নিউরাল নেটওয়ার্ক স্ট্রাকচার যেখানে ডেটা শুধুমাত্র একদিকেই প্রবাহিত হয় — ইনপুট থেকে আউটপুট পর্যন্ত। এতে কোনো লুপ বা রিকারশন থাকে না।

### গঠন:

* Input Layer → Hidden Layer(s) → Output Layer
* প্রতিটি লেয়ারে নিউরন থাকে এবং তারা পরবর্তী লেয়ারের সব নিউরনের সঙ্গে সংযুক্ত থাকে (fully connected)

### কিভাবে কাজ করে:

* ইনপুট ফিচার আসে
* হিডেন লেয়ারগুলো একে একে তথ্য প্রসেস করে
* শেষ আউটপুট আসে প্রেডিকশন বা ক্লাসিফিকেশনের জন্য

### ব্যবহার:

* ইমেইল স্প্যাম ফিল্টার
* হাউজ প্রাইস রিগ্রেশন
* বেসিক ক্লাসিফিকেশন টাস্ক

---

## ২. Convolutional Neural Network (CNN)

### সংজ্ঞা:

CNN এমন একটি নিউরাল নেটওয়ার্ক যা মূলত image ও video ডেটা প্রসেসিংয়ের জন্য ব্যবহৃত হয়। এতে convolutional লেয়ার থাকে যা ছবি থেকে ফিচার এক্সট্র্যাক্ট করতে পারে।

### গঠন:

* Convolutional Layer → Activation Layer → Pooling Layer → Fully Connected Layer
* Conv লেয়ার ইমেজের ছোট অংশ স্ক্যান করে গুরুত্বপূর্ণ ফিচার তুলে ধরে

### কিভাবে কাজ করে:

* ইমেজ ইনপুট হয়
* Conv লেয়ারে ফিচার ডিটেক্ট হয় (edge, corner ইত্যাদি)
* Pooling লেয়ার ডেটাকে কম্প্রেস করে
* Fully connected লেয়ারে ফাইনাল প্রেডিকশন হয়

### ব্যবহার:

* ফেস রিকগনিশন (Facebook tagging)
* সেলফ ড্রাইভিং কার (Road lane detection)
* মেডিকেল ইমেজ এনালাইসিস (Tumor detection)

---

## ৩. Recurrent Neural Network (RNN)

### সংজ্ঞা:

RNN টাইম-সিরিজ বা সিকোয়েন্স ডেটা হ্যান্ডল করার জন্য ব্যবহৃত হয়। এটি পূর্ববর্তী ইনপুটের তথ্য মনে রাখতে পারে এবং একই নিউরন বারবার ব্যবহার হয়।

### গঠন:

* একটি লুপযুক্ত স্ট্রাকচার যেখানে প্রতিটি ধাপে তথ্য পূর্ববর্তী স্টেট থেকে ইনহেরিট করে

### কিভাবে কাজ করে:

* ইনপুট সিকোয়েন্স ধাপে ধাপে দেয়া হয় (যেমন শব্দ বা সময়)
* প্রতিটি সময়ের ইনপুট আগের স্টেটের সাথে যুক্ত হয়ে আউটপুট দেয়
* এটা sequential data modelling-এর জন্য উপযুক্ত

### ব্যবহার:

* ভয়েস টু টেক্সট
* ভাষা মডেল (Next word prediction)
* স্টক প্রাইস প্রেডিকশন

---

## ৪. Long Short-Term Memory (LSTM)

### সংজ্ঞা:

LSTM হল RNN এর একটি উন্নত ফর্ম যা long-term dependency হ্যান্ডেল করতে পারে। এতে cell state এবং gate mechanisms ব্যবহার করা হয়।

### গঠন:

* Input Gate, Forget Gate, Output Gate
* Cell state লং-টার্ম ইনফরমেশন বহন করে

### কিভাবে কাজ করে:

* Input gate বর্তমান ইনফরমেশন retain করবে কিনা তা ঠিক করে
* Forget gate আগের তথ্য রাখবে না ফেলবে তা ঠিক করে
* Output gate আউটপুটে কী যাবে তা নির্ধারণ করে

### ব্যবহার:

* মিউজিক জেনারেশন
* ভিডিও ক্যাপশনিং
* ভাষা অনুবাদ

---

## ৫. Transformer Network

### সংজ্ঞা:

Transformer এমন একটি নিউরাল নেটওয়ার্ক আর্কিটেকচার যা attention mechanism ব্যবহার করে। এটি একসাথে পুরো ইনপুট প্রসেস করতে পারে, একধরনের parallelism সমর্থন করে।

### গঠন:

* Encoder এবং Decoder ব্লক
* Multi-head Attention, Feed Forward Layer, Positional Encoding

### কিভাবে কাজ করে:

* Encoder ইনপুট সিকোয়েন্স থেকে context-aware representation বানায়
* Decoder সেই representation ব্যবহার করে আউটপুট তৈরি করে
* Attention লেয়ার কোন ইনপুট কতটা গুরুত্বপূর্ণ সেটা বুঝে নেয়

### ব্যবহার:

* GPT, BERT, ChatGPT
* Machine Translation (Google Translate)
* Text Summarization, Question Answering

---

## উপসংহার

এই পাঁচটি নিউরাল নেটওয়ার্ক টাইপ বিভিন্ন ধরণের ডেটা ও সমস্যার জন্য উপযোগী।

* FNN সাধারণ কাজে
* CNN ইমেজ ও ভিশন টাস্কে
* RNN ও LSTM টাইম-সিরিজ ও ভাষাগত সমস্যায়
* Transformer অত্যাধুনিক NLP ও ট্রান্সফার লার্নিং-এ অনন্য ভূমিকা রাখছে।

এই স্ট্রাকচারগুলো বোঝা Deep Learning শেখার ভিত্তি তৈরি করে।
