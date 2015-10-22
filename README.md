# Music-Transcription
### Todos
- Work on downsampling the data--lines 20-22 of handlewav.py; Try using scipy.signal.resample
- Implement functions to load and preprocess the data
    - Consider first preprocessing all the data and storing it as matrix files, and using those for training
- Test network
    - Code for training (SGD) and running the network, implementing loss functions
        - Decrease learning rate linearly

*Resolved*
- ~~Look into cPickle import issue, it only occurs when importing from six.moves, so perhaps try uninstalling and reinstalling six (get an earlier version)~~
- ~~Uninstall and reinstall bleeding-edge theano (made some changes in this one, best to just get a fresh copy)~~
