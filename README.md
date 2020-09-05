# American Express Challenge 2020

## Artificially intelligent ChatBot for travel recommendation (Software)

### :wrench: Installation of packages via python3 in Terminal prompt

<b>To create a python virtual environment for the project:</b>
python3 -m venv .venv 
source .venv/bin/activate

<b>Install packages via pip:</b>
pip install -r requirements.txt

<b><i>The project has been coded using python3.8.5 and works with the version specified in the requirements</i></b>

### :horse_racing: Training the Deep Learning models

The codes to train the models (FCNN, LSTM, RNNLM and BERT) can be found in the python file <i>Training.py</i>, a notebook version is saved in <i>Training_notebook.ipynb</i> and a universally readable htlm file is saved in <i>Training_notebook.html</i>.

I do recommend to use Visual Studio Code and its "Python Interactive" shell to run the Training.py file in order to train the model.

You can also use the notebook and <b>jupyter notebook</b> to run the trainings.

If you don't have the patience to wait for an eight-hour long training, simply download the <a href="">models' weights</a> and copy-paste them into the <i>model/</i> folder.

### :robot: Interact with the DL models via a ChatBot

### Slack Bot implementation :computer:

The <a url="https://github.com/slackapi/Slack-Python-Onboarding-Tutorial">following github repository</a> explains how you can launch your bot. You just need to replace his python script by the one we implemented here <i>mainbot.py</i>. To sum-up the steps, you need to export the credentials of your Slack app, then create a HTTP tunnel from your local server to a ngrok web server, then you give Slack the url of this exposed localhost and finally you run your python code.

python mainbot.py

Then you go in Slack and interact with the Chatbot in a dedicated channel.

For the American Express Challenge 2020, we will keep our Slack channel open for at least 12 days (until 09/17/2020) so that you can interact with our :robot: <b>amexbot</b>.

To join the dedicated Slack Team, click <a href="https://join.slack.com/t/dorebeen/shared_invite/zt-gne8osn9-5YUofOA9m7fZwy1roaePAA">here</a>.

### Chatbot on your own terminal screen :tv:

python main.py

Keep in mind the 8 labels and ask your questions to see how the models react to them:
The model only handles questions concerning one country <b>Beijing</b>
<ul>
    <li>Greeting</li>
    <li>Historical</li>
    <li>Food</li>
    <li>Animals</li>
    <li>Cruise</li>
    <li>Night tour</li>
    <li>Relaxation</li>
    <li>Show concerts</li>
</ul>

### To do implementation on a website :bookmark_tabs:

## The code behind -- Further development

Please look at the code in <i>ai_bot.py</i> to understand the preprocessing of the data and the construction of the architectures of the models used. Some models haven't been presented in the frontend of the software for clarity purposes. 

Further developments can be done concerning the implementation of country detection coupled with intent detection in order to give more precise information and tackle a broader public. More features could be studied: the country, the price, the preference... A clearer mapping between theses features and travel deals would be key for the success of such a Chatbot.

Certain limitations could also be adressed. We used a pretty big hand-typed dataset, but it is not big enough to avoid biases, since we come from a certain background, our way of speaking and the questions we think about are limited compared to what people from all over the world would think of and type to the chatbot. This curse of dimensionality can be tackled using state-of-the-art models like Bert but also by diverfying the way the sentences are written.

## Contributing
 
1. Fork it!
2. Create your feature branch: `git checkout -b my-new-feature`
3. Commit your changes: `git commit -am 'Add some feature'`
4. Push to the branch: `git push origin my-new-feature`
5. Submit a pull request :smile:
 
## History
 
Version 0.1 (2020-09-05) - adding Devise and core Rails functionality
 
## Credits
 
Lead Developer - Emmanuel Ren (@eren125)
Main contributor - Dabeen Oh (@dabeenoh)
Main contributor - Soyeong Bak (@111570)

## License
 
The MIT License (MIT)

Copyright (c) 2020 Emmanuel Ren

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.