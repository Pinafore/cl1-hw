
Big Pictuure
=====

While we've trained models from scratch, often the best place to start is with
an existing model.  In this homework, we're going to take a large(ish) model
and fine tune it so that it can do the same take that you saw in your feature
engineering homework: predicting whether an answer to a question is correct or
not.

This homework is worth 15 points.

What you Have to Do
=====

As usual, you may need to install athe required Python environment (there's a lot this time).

    ./venv/bin/pip3 install -r requirements.txt

After understanding the code, you can get down to coding:

* To make sure that this is actually efficient, you will need to freeze the
  model's original parameters.  Set the `requires_grad` for all of the base
  model parameters to `False`.  You need to do this in the
  `initialize_base_model` code.  *Do not overlook this, as it will work
  without this change but will be very slow*.  The first time you run the code will take a little bit longer because it needs
to download the DistillBERT model.


     ./venv/bin/python3 lorabert_buzzer.py 
     config.json: 100%|████████████████████████████████████████████| 483/483 [00:00<00:00, 7.18MB/s]
     model.safetensors: 100%|████████████████████████████████████| 268M/268M [00:04<00:00, 64.1MB/s]


This will go faster afterward.


* You will need to define the parameter matrices for the LoRA layer in the
  `LoRALayer` class `__init__` function and then use them to compute a delta
  in the `forward` function.

* Likewise, you will need to add a `LoRALayer` component to the `LinearLoRA`
  class and change its `forward` function to use that delta in its forward
  function.  (I realize this could have been one class, but this makes testing
  easier... it also makes it possible to have more LoRA adaptations beyond
  adapting just linear layers.)

* Now that we have the tools for changing some layers, we now need to add them
  to the frozen model we created in `initialize_base_model` in the `add_lora`
  function.  You will probably want to create a (partial
  object)[https://docs.python.org/3/library/functools.html#partial-objects].  

* You shouldn't need to change anything in `LoRABertBuzzer`, it should run the
  training for you and prepare the data.

* Run adaptation on some data (use `limit` if you don't have a GPU).  This is
  more of a proof-of-concept, and you don't need great accuracy to satisfy the
  requirements of the homework (but loss should go down and accuracy should
  improve with more data).


Extra Credit
======

If you complete the extra credit, please submit `analysis.pdf` describing what
you did and how you evaluated whether it worked well.

* [Up to 10 Points] Improve the performance of the fine-tuned system.  There
  are vey easy ways to do this: we are forming the `text` field of the
  examples in a fairly naive way.  We could add more information or format it
  better.  A more involved (but likely better) is to further extend the model
  to better encode additional floating point features (like you did in the
  feature engineering homework).

* [Up to 5 Points] Experiment with what layers are most necessary for the best
  improvements and test values of alpha and rank that work best (you cannot
  use tiny datasets for this, unfortunately, so this requires a GPU, probably
  ... not a great one, as any GPU will likely be fine).  Make sure in addition to any accuracy / buzz ratio numbers you provide you also count the number of parameters.

* [Up to 3 Points] The training code in `train` are taken directly from the
  Huggingface examples and I didn't think too much about them.  It's not clear
  that they're a good fit for the data.  Can you find something substantially
  better?  (Keeping the model / adaptation / etc. constant.) 


FAQ
========

*Q: Why do the unit tests use "encoder" but the template code uses "transformer"?*

*A:* The unit tests are using a "real" (but very tiny) BERT, while the template code is using DistilBERT (but much larger).  They are packaged slightly differently.
