# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
from collections import Counter
from multiprocessing import Pool

import torch
from fairseq import utils
from fairseq.data import data_utils
from fairseq.file_chunker_utils import Chunker, find_offsets
from fairseq.file_io import PathManager
from fairseq.tokenizer import tokenize_line


class Dictionary:
    """A mapping from symbols to consecutive integers"""

    def __init__(
        self,
        *,  # begin keyword-only arguments
        bos="<s>",
        pad="<pad>",
        eos="</s>",
        unk="<unk>",
        sep='<sep>', 
        isep='<isep>',
        extra_special_symbols=None,
    ):
        self.bos_word, self.unk_word, self.pad_word, self.eos_word = bos, unk, pad, eos
        self.sep_word = sep
        self.isep_word = isep
        self.symbols = []
        self.count = []
        self.indices = {}
        self.add_symbol('<Lua heritage>')
        #self.bos_index = self.add_symbol(bos)
        self.pad_index = self.add_symbol(pad)
        self.eos_index = self.add_symbol(eos)
        self.unk_index = self.add_symbol(unk)
        self.sep_index = self.add_symbol(sep)
        self.isep_index = self.add_symbol(isep)
        if extra_special_symbols:
            for s in extra_special_symbols:
                self.add_symbol(s)
        self.nspecial = len(self.symbols)
        # print('N special symbols: ', self.symbols)
        # print("sep",self.sep_index)
        # print("isep",self.isep_index)
        # print("pad",self.pad_index)
        # print("unk",self.unk_index)

    def __eq__(self, other):
        return self.indices == other.indices

    def __getitem__(self, idx):
        if idx < len(self.symbols):
            return self.symbols[idx]
        return self.unk_word

    def get_count(self, idx):
        return self.count[idx]

    def __len__(self):
        """Returns the number of symbols in the dictionary"""
        return len(self.symbols)

    def __contains__(self, sym):
        return sym in self.indices

    def index(self, sym):
        """Returns the index of the specified symbol"""
        assert isinstance(sym, str)
        if sym in self.indices:
            return self.indices[sym]
        return self.unk_index

    def string(
        self,
        tensor,
        bpe_symbol=None,
        escape_unk=False,
        extra_symbols_to_ignore=None,
        unk_string=None,
        include_eos=False,
        separator=" ",
        tgt_dict=None,
        escape_pad=False,
    ):
        """Helper for converting a tensor of token indices to a string.

        Can optionally remove BPE symbols or escape <unk> words.
        """
        # if torch.is_tensor(tensor) and tensor.dim() == 2:
        #     return "\n".join(
        #         self.string(
        #             t,
        #             bpe_symbol,
        #             escape_unk,
        #             extra_symbols_to_ignore,
        #             include_eos=include_eos,
        #         )
        #         for t in tensor
        #     )

        # extra_symbols_to_ignore = set(extra_symbols_to_ignore or [])
        # if not include_eos:
        #     extra_symbols_to_ignore.add(self.eos())

        # def token_string(i):
        #     if i == self.unk():
        #         if unk_string is not None:
        #             return unk_string
        #         else:
        #             return self.unk_string(escape_unk)
        #     else:
        #         return self[i]

        # if hasattr(self, "bos_index"):
        #     extra_symbols_to_ignore.add(self.bos())

        # sent = separator.join(
        #     token_string(i)
        #     for i in tensor
        #     if utils.item(i) not in extra_symbols_to_ignore
        # )

        # return data_utils.post_process(sent, bpe_symbol)

        if torch.is_tensor(tensor) and tensor.dim() == 2:
            return '\n'.join(self.string(t, tgt_dict=tgt_dict, escape_pad=escape_pad) for t in tensor)

        def token_string(i,idx,sep_id=None):
            if i == self.unk():
                return self.unk_string(escape_unk)
            elif tgt_dict is not None:
                word=tgt_dict[i] if ((sep_id > 0) and (idx > sep_id)) else self[i]
                return word
            elif escape_pad and i==self.pad():
                return ''
            else:
                return self[i]
        
        token_seq = [int(x) for x in tensor]
        #print("before becoming a string it looks like",token_seq)
        sep_id = token_seq.index(self.sep_index) if (self.sep_index in token_seq) else -1
        sent = ' '.join(token_string(i,idx,sep_id=sep_id) \
            for idx,i in enumerate(tensor) if i != self.eos())
        
        #print("Inside dictionary line 81, sent without bpe processing",sent)
        return data_utils.post_process(sent, bpe_symbol)

    def unk_string(self, escape=False):
        """Return unknown string, optionally escaped as: <<unk>>"""
        if escape:
            return "<{}>".format(self.unk_word)
        else:
            return self.unk_word

    # def add_symbol(self, word, n=1, overwrite=False):
    #     """Adds a word to the dictionary"""
    #     if word in self.indices and not overwrite:
    #         idx = self.indices[word]
    #         self.count[idx] = self.count[idx] + n
    #         return idx
    #     else:
    #         idx = len(self.symbols)
    #         self.indices[word] = idx
    #         self.symbols.append(word)
    #         self.count.append(n)
    #         return idx

    def add_symbol(self, word, n=1):
        """Adds a word to the dictionary"""
        if word in self.indices:
            idx = self.indices[word]
            self.count[idx] = self.count[idx] + n
            return idx
        else:
            idx = len(self.symbols)
            self.indices[word] = idx
            self.symbols.append(word)
            self.count.append(n)
            return idx

    # def update(self, new_dict):
    #     """Updates counts from new dictionary."""
    #     for word in new_dict.symbols:
    #         idx2 = new_dict.indices[word]
    #         if word in self.indices:
    #             idx = self.indices[word]
    #             self.count[idx] = self.count[idx] + new_dict.count[idx2]
    #         else:
    #             idx = len(self.symbols)
    #             self.indices[word] = idx
    #             self.symbols.append(word)
    #             self.count.append(new_dict.count[idx2])

    def update(self, new_dict):
        """Updates counts from new dictionary."""
        for word in new_dict.symbols:
            idx2 = new_dict.indices[word]
            if word in self.indices:
                idx = self.indices[word]
                self.count[idx] = self.count[idx] + new_dict.count[idx2]
            else:
                idx = len(self.symbols)
                self.indices[word] = idx
                self.symbols.append(word)
                self.count.append(new_dict.count[idx2])

    def finalize(self, threshold=-1, nwords=-1, padding_factor=8):
        """Sort symbols by frequency in descending order, ignoring special ones.

        Args:
            - threshold defines the minimum word count
            - nwords defines the total number of words in the final dictionary,
                including special symbols
            - padding_factor can be used to pad the dictionary size to be a
                multiple of 8, which is important on some hardware (e.g., Nvidia
                Tensor Cores).
        """
        if nwords <= 0:
            nwords = len(self)

        new_indices = dict(zip(self.symbols[:self.nspecial], range(self.nspecial)))
        new_symbols = self.symbols[:self.nspecial]
        new_count = self.count[:self.nspecial]

        c = Counter(dict(zip(self.symbols[self.nspecial:], self.count[self.nspecial:])))
        for symbol, count in c.most_common(nwords - self.nspecial):
            if count >= threshold:
                new_indices[symbol] = len(new_symbols)
                new_symbols.append(symbol)
                new_count.append(count)
            else:
                break

        threshold_nwords = len(new_symbols)
        if padding_factor > 1:
            i = 0
            while threshold_nwords % padding_factor != 0:
                symbol = 'madeupword{:04d}'.format(i)
                new_indices[symbol] = len(new_symbols)
                new_symbols.append(symbol)
                new_count.append(0)
                i += 1
                threshold_nwords += 1

        assert len(new_symbols) % padding_factor == 0
        assert len(new_symbols) == len(new_indices)

        self.count = list(new_count)
        self.symbols = list(new_symbols)
        self.indices = new_indices

    def pad_to_multiple_(self, padding_factor):
        """Pad Dictionary size to be a multiple of *padding_factor*."""
        if padding_factor > 1:
            i = 0
            while len(self) % padding_factor != 0:
                symbol = "madeupword{:04d}".format(i)
                self.add_symbol(symbol, n=0)
                i += 1

    def bos(self):
        """Helper to get index of beginning-of-sentence symbol"""
        return self.bos_index

    def pad(self):
        """Helper to get index of pad symbol"""
        return self.pad_index

    def eos(self):
        """Helper to get index of end-of-sentence symbol"""
        return self.eos_index

    def unk(self):
        """Helper to get index of unk symbol"""
        return self.unk_index

    def sep(self):
        """Helper to get index of sep symbol"""
        return self.sep_index

    def isep(self):
        """Helper to get index of isep symbol"""
        return self.isep_index

    @classmethod
    def load(cls, f, ignore_utf_errors=False):
        """Loads the dictionary from a text file with the format:

        ```
        <symbol0> <count0>
        <symbol1> <count1>
        ...
        ```
        """
        if isinstance(f, str):
            try:
                if not ignore_utf_errors:
                    with open(f, 'r', encoding='utf-8') as fd:
                        return cls.load(fd)
                else:
                    with open(f, 'r', encoding='utf-8', errors='ignore') as fd:
                        return cls.load(fd)
            except FileNotFoundError as fnfe:
                raise fnfe
            except UnicodeError:
                raise Exception("Incorrect encoding detected in {}, please "
                                "rebuild the dataset".format(f))

        d = cls()
        lines = f.readlines()
        indices_start_line = d._load_meta(lines)
        for line in lines[indices_start_line:]:
            idx = line.rfind(' ')
            if idx == -1:
                raise ValueError("Incorrect dictionary format, expected '<token> <cnt>'")
            word = line[:idx]
            count = int(line[idx + 1:])
            d.indices[word] = len(d.symbols)
            d.symbols.append(word)
            d.count.append(count)
            # print(len(d))
            # exit()
        return d

    def add_from_file(self, f):
        """
        Loads a pre-existing dictionary from a text file and adds its symbols
        to this instance.
        """
        if isinstance(f, str):
            try:
                with open(PathManager.get_local_path(f), "r", encoding="utf-8") as fd:
                    self.add_from_file(fd)
            except FileNotFoundError as fnfe:
                raise fnfe
            except UnicodeError:
                raise Exception(
                    "Incorrect encoding detected in {}, please "
                    "rebuild the dataset".format(f)
                )
            return

        lines = f.readlines()
        indices_start_line = self._load_meta(lines)

        for line in lines[indices_start_line:]:
            try:
                line, field = line.rstrip().rsplit(" ", 1)
                if field == "#fairseq:overwrite":
                    overwrite = True
                    line, field = line.rsplit(" ", 1)
                else:
                    overwrite = False
                count = int(field)
                word = line
                if word in self and not overwrite:
                    raise RuntimeError(
                        "Duplicate word found when loading Dictionary: '{}'. "
                        "Duplicate words can overwrite earlier ones by adding the "
                        "#fairseq:overwrite flag at the end of the corresponding row "
                        "in the dictionary file. If using the Camembert model, please "
                        "download an updated copy of the model file.".format(word)
                    )
                self.add_symbol(word, n=count, overwrite=overwrite)
            except ValueError:
                raise ValueError(
                    f"Incorrect dictionary format, expected '<token> <cnt> [flags]': \"{line}\""
                )

    # def _save(self, f, kv_iterator):
    #     if isinstance(f, str):
    #         PathManager.mkdirs(os.path.dirname(f))
    #         with PathManager.open(f, "w", encoding="utf-8") as fd:
    #             return self.save(fd)
    #     for k, v in kv_iterator:
    #         print("{} {}".format(k, v), file=f)

    def _save(self, f, kv_iterator):
        if isinstance(f, str):
            os.makedirs(os.path.dirname(f), exist_ok=True)
            with open(f, 'w', encoding='utf-8') as fd:
                return self.save(fd)
        for k, v in kv_iterator:
            print('{} {}'.format(k, v), file=f)

    def _save_nspecial(self, f, kv_iterator):
        if isinstance(f, str):
            os.makedirs(os.path.dirname(f), exist_ok=True)
            with open(f, 'w', encoding='utf-8') as fd:
                return self.save_nspecial(fd)
        for k, v in kv_iterator:
            print('{} {}'.format(k, v), file=f)

    def _get_meta(self):
        return [], []

    def _load_meta(self, lines):
        return 0

    def save(self, f):
        """Stores dictionary into a text file"""
        ex_keys, ex_vals = self._get_meta()
        self._save(
            f,
            zip(
                ex_keys + self.symbols[self.nspecial :],
                ex_vals + self.count[self.nspecial :],
            ),
        )

    def save_nspecial(self, f):
        """Stores dictionary into a text file"""
        ex_keys, ex_vals = self._get_meta()
        self._save_nspecial(f, zip(ex_keys + self.symbols[:self.nspecial], ex_vals + self.count[:self.nspecial]))

    def dummy_sentence(self, length):
        t = torch.Tensor(length).uniform_(self.nspecial + 1, len(self)).long()
        t[-1] = self.eos()
        return t

    def encode_line(
        self,
        line,
        line_tokenizer=tokenize_line,
        add_if_not_exist=True,
        consumer=None,
        append_eos=True,
        reverse_order=False,
        tgt_dict=None
    ) -> torch.IntTensor:
        # words = line_tokenizer(line)
        # if reverse_order:
        #     words = list(reversed(words))
        # nwords = len(words)
        # ids = torch.IntTensor(nwords + 1 if append_eos else nwords)

        # for i, word in enumerate(words):
        #     if add_if_not_exist:
        #         idx = self.add_symbol(word)
        #     else:
        #         idx = self.index(word)
        #     if consumer is not None:
        #         consumer(word, idx)
        #     ids[i] = idx
        # if append_eos:
        #     ids[nwords] = self.eos_index
        # return ids
        words = line_tokenizer(line)
        if reverse_order:
            words = list(reversed(words))
        nwords = len(words)
        ids = torch.IntTensor(nwords + 1  if append_eos else nwords)
        if tgt_dict is not None:
            sep_idx = words.index('<sep>') if '<sep>' in words else -1

        # print("sep idx",sep_idx)

        for i, word in enumerate(words):
            if add_if_not_exist:
                idx = self.add_symbol(word)
            elif tgt_dict is None:
                idx = self.index(word)                    
            else:
                idx = self.index(word) if (i < sep_idx or sep_idx < 0) else tgt_dict.index(word)

                # print("word:",word," index:",idx)
            if consumer is not None:
                consumer(word, idx)
            ids[i] = idx
        if append_eos:
            ids[nwords] = self.eos_index
        return ids

    @staticmethod
    def _add_file_to_dictionary_single_worker(
        filename,
        tokenize,
        eos_word,
        start_offset,
        end_offset,
    ):
        counter = Counter()
        with Chunker(filename, start_offset, end_offset) as line_iterator:
            for line in line_iterator:
                for word in tokenize(line):
                    counter.update([word])
                counter.update([eos_word])
        return counter

    @staticmethod
    def add_file_to_dictionary(filename, dict, tokenize, num_workers):
        def merge_result(counter):
            for w, c in sorted(counter.items()):
                dict.add_symbol(w, c)

        local_file = PathManager.get_local_path(filename)
        offsets = find_offsets(local_file, num_workers)
        if num_workers > 1:
            chunks = zip(offsets, offsets[1:])
            pool = Pool(processes=num_workers)
            results = []
            for (start_offset, end_offset) in chunks:
                results.append(
                    pool.apply_async(
                        Dictionary._add_file_to_dictionary_single_worker,
                        (
                            local_file,
                            tokenize,
                            dict.eos_word,
                            start_offset,
                            end_offset,
                        ),
                    )
                )
            pool.close()
            pool.join()
            for r in results:
                merge_result(r.get())
        else:
            merge_result(
                Dictionary._add_file_to_dictionary_single_worker(
                    local_file, tokenize, dict.eos_word, offsets[0], offsets[1]
                )
            )


class TruncatedDictionary(object):
    def __init__(self, wrapped_dict, length):
        self.__class__ = type(
            wrapped_dict.__class__.__name__,
            (self.__class__, wrapped_dict.__class__),
            {},
        )
        self.__dict__ = wrapped_dict.__dict__
        self.wrapped_dict = wrapped_dict
        self.length = min(len(self.wrapped_dict), length)

    def __len__(self):
        return self.length

    def __getitem__(self, i):
        if i < self.length:
            return self.wrapped_dict[i]
        return self.wrapped_dict.unk()
