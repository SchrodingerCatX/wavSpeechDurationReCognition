# wavSpeechDurationReCognition
By using OpenAI's Whisper language model, after inputting a wav language file, the duration of English, Chinese and Cantonese speech can be recognized.

model.safetensors is large, you need to download it and add to your project by yourself.


1. A 2S window with 70% overlap, set the window size to 3, and perform smoothing based on the median value. Take the two values before and after for a preliminary filtering judgment.

2. Special processing is applied to the starting and ending data to ensure that all are processed with 3 values. The result is that the first and second elements use the same 3 window values, and the last element and the second-to-last element use the same 3 window values.

3. Special processing: If the 3 values in the window correspond to 3 different languages, the language of the first value in the window is selected by default, considering a language shift has occurred. It is possible to take either the previous or the next value, but here the previous one is taken, which has little impact on the accuracy of the calculation duration.

4. Further optimization of the results:
Single-point smoothing: For each value in smooth_codes, check its adjacent values before and after. If the current value is different from the adjacent values and the adjacent values are the same, modify the current value to the adjacent value. The first and last elements are not processed.
Double-point smoothing: Find all consecutive equal values and compare their two neighbors before and after. If the current two values are different from the two previous neighbors and the two previous neighbors are the same, modify the current two values to the neighbor values. The first and last elements are not processed.
Three-point smoothing: Find all consecutive equal values and compare their three neighbors before and after. If the current three values are different from the three previous neighbors and the three previous neighbors are the same as the three following neighbors, modify the current three values to the neighbor values. The first and last elements are not processed.
Keep the code structure unchanged and only optimize the calculation logic of smooth_codes to adjust the values more smoothly. 

5. Noise Optimization:
Continue with special processing, as the mutation points should be where language transitions occur; otherwise, mutations would not happen. If a value is different from both its preceding and following neighbors, it is considered to be noise generated during the switch from the language of the preceding neighbor to that of the following neighbor. In such cases, it should be proactively replaced with the value of the following neighbor.
Search for all consecutive N equal values and compare their N preceding and following neighbor values. If the N preceding neighbor values are equal, the N following neighbor values are equal, the N preceding neighbor values are not equal to the N following neighbor values, and the current N values are different from both the preceding and following N neighbor values, then modify the current N values to the value of the following neighbor. 

6. Head and Tail Optimization:
Check the first N values in the head one by one, where N ranges from 1 to half of the language with the least occurrence. If the current value is different from the subsequent N neighboring values, modify the current value to the neighboring value.
Check the last N values in the tail one by one, where N ranges from 1 to half of the language with the least occurrence. If the current value is different from the preceding N neighboring values, modify the current value to the neighboring value.