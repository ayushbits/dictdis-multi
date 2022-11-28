## Very Important Note

If you would like to train while ingesting random constraints for non-dcitionary sentences, then you need to ensure that line 340 in language_pair_dataset.py is commented :
``` 
elif self.sep_idx in self.src[index]: # this is added for inference by Ayush
src_item = self.src[index]
```

- During inference, this line should be uncommented to prevent going inside get_rand_cons().