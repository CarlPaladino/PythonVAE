Experimental progress:
Initially tested fully AI generated version to see how a VAE functions at all and how the variables interact.
    Had AI implement CNN functionality

Handwrote my own version using loops and minimal matrix mathematics/patching. Tested 5 samples over 50 epochs initially to see what it produced, it started converging as I could see from the total loss and reconstructed examples
    Then I had AI write improvements to it using the same architecture and minor changes such as taking advantage of patching values and matrix multiplication where applicable and it produced the exact same result on the initial 5 seeded samples after 50 epochs. After validating that it would produce the same output on the same data after same amount of epochs, then I could confidently allow it to run as my own CNN-VAE. Tested on larger dataset to see how it would converge and perform.