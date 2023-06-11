# Deep Sequoia corpus version 9.2

October 2020

## Format documentation

The format is an extended version of the [CoNLL-U plus](https://universaldependencies.org/ext-format.html) format:

- columns 1 to 10 encode morphology and syntax in a CoNLL-U inspired format, adapted to encode graphs
  -- the syntactic columns (7 and 8) encode both the surface dependency tree and the deep syntactic graph
  -- named the "[deep_and_surf](README-format.md#" format
- column 11 encodes the PARSEME-FR annotation of named entities and multi-word expressions
- column 12 contains the FRSEMCOR semantic annotation on nouns

 * [Deep and surf format for syntax](README-format.md#deep-and-surf-dependency-format)
 * [PARSEME-FR MWE annotation format](README-format.md#parseme-fr-annotations)
 * [FRSEMCOR semantic annotation format](README-format.md#frsemcor-annotations)

## Deep-and-surf dependency format
For a global view of different dependency formats, see [here](https://deep-sequoia.inria.fr/process/).

The deep-and-surf format is a compact representation containing **BOTH** the (surface) dependency tree
and the deep dependency graph.

For each token t, columns 7 and 8 encode one or several arcs in which t is the dependent:
this is represented by column 7 containing several labels, and column 8 containing several governor ids.

The number of labels in column 7 is the same as the number of governors in column 8 and they are interpreted in parallel:
first label in column 7 corresponds to first governor in column 8, etc.

For instance, the noun *réunion* in the example below depends on tokens 5 and 2, with labels "suj:obj" and "D:suj:obj" respectively.

```
# sent_id = annodis.er_00449
# text = Une prochaine réunion est prévue mardi
1	Une	un	D	DET	g=f|n=s|s=ind	3	det	_	_
2	prochaine	prochain	A	ADJ	g=f|n=s|s=qual	3	mod	_	_
3	réunion	réunion	N	NC	g=f|n=s|s=c	5|2	suj:obj|D:suj:suj	_	_
4	est	être	V	V	dl=être|m=ind|n=s|p=3|t=pst|void=y	5	S:aux.pass	_	_
5	prévue	prévoir	V	VPP	diat=passif|dl=prévoir|dm=ind|g=f|m=part|n=s|t=past	0	root	_	_
6	mardi	mardi	N	NC	g=m|n=s|s=c	5	mod	_	_
```

A given arc belongs to either
- the surface tree only (**prefix `S:`**),
- the deep syntactic graph only (**prefix `D:`**),
- or to both (**no prefix**).

 * relations subject to diathesis alternations contain a double label xxx:yyy, where xxx stands for the "final" grammatical function, and yyy stands for the canonical grammatical function. (ex: `suj:obj` stands for final subject and canonical object)
 * a special feature `void=y` is used for tokens not belonging to the deep graph: those who have no incoming arc in the deep graph (e.g. token 4 above).
One can choose to ignore these tokens, or to attach them to the fictive root with a dummy label.


### How to obtain the surf dependency tree:

The surf format is obtained by removing relations prefixed by `D:` and by taking the first part of double labels ("final grammatical functions").

### How to obtain the deep syntactic graph:

The deep format is obtained by removing relations prefixed by `S:` and by taking the second part of double labels.

For some applications, it might be useful though to also use the final labels
(namely it might be useful to know a NP is the subject of a passive verb).


## PARSEME-FR annotations:

See the [PARSEME-FR format description](https://gitlab.lis-lab.fr/PARSEME-FR/PARSEME-FR-public/wikis/Corpus-format-description)
for more info about MWE and named entities encoding.


## FRSEMCOR annotations:

Single nouns, nominal MWEs and named entities are annotated with a semantic class (a "supersense").

See the [FRSEMCOR format description](https://github.com/FrSemCor/FrSemCor/blob/master/fr_semcor_format).
