# Deep Sequoia corpus version 9.2
October 2020

 * [Summary](README-distrib.md#summary)
 * [Links](README-distrib.md#links)
 * [Licence](README-distrib.md#licence)
 * [History of the corpus](README-distrib.md#history-of-the-corpus)
 * [References](README-distrib.md#references)
 * [Content](README-distrib.md#content)
 * [Dependency formats](README-distrib.md#dependency-formats)
 * [Constituency format](README-distrib.md#constituency-format)
 * [Appendix](README-distrib.md#appendix)


## Summary
The corpus contains 3,099 French sentences, from Europarl, Est Republicain newspaper,
French Wikipedia and European Medicine Agency, with the following manual annotations:

* Parts-of-speech and morphological features
* Grammatical compound words (merged as one token before version **8.0** and with multi-tokens linked by `dep_cpd` relation since version **8.0**)
* Surface syntax: dependencies and constituents (constituents were not updated after version **7.0**)
* Deep syntax (dependencies only, since version **7.0**)
* Multi-Word Expressions and Named Entities (since **9.0**)
* Semantic classes ("supersenses") for nouns (since **9.1**)

## Links
* Contact: deep-sequoia@inria.fr
* Website: [deep-sequoia.inria.fr](https://deep-sequoia.inria.fr)
* Annotation guide for syntax (in French): [deep-sequoia-guide.pdf](https://deep-sequoia.inria.fr/deep-sequoia-guide.pdf)
* Annotation guide for Multi-Word expressions and named entities: [PARSEME-FR guide](https://gitlab.lis-lab.fr/PARSEME-FR/PARSEME-FR-public/wikis/Guide-annotation-PARSEME_FR-chapeau)
* Information and Annotation guide for supersenses : [FRSEMCOR website](https://frsemcor.github.io/FrSemCor/)

## Licence
The corpus is freely available under the free licence [LGPL-LR
(Lesser General Public License For Linguistic Resources)](https://deep-sequoia.inria.fr/licence/).

## History of the corpus
The Sequoia corpus was first manually annotated for part-of-speech and phrase-structure, and automatically converted to surface syntactic dependency trees.
(Candito and Seddah, 2012a).
The phrase-structure annotation follows mainly the French Treebank guidelines
(http://www.llf.cnrs.fr/Gens/Abeille/French-Treebank-fr.php),
modified in the context of conversion to dependencies:

   - prepositions that dominate a infinitival VP do project a PP
   - any sentence introduced by a complementizer (CS tag) is grouped into a Sint constituent


A further step of manual annotation was carried out, aiming at correcting the governor of extracted elements:
in case of long-distance dependency, the automatic conversion from constituents to dependencies picks out a wrong governor for the extracted element.
These were manually corrected, leading to a few non-projective links.
(Candito and Seddah, 2012b).

**Deep syntactic dependencies:** Then a collaboration started in 2013 between the Alpage and Sémagramme teams,
to obtain DEEP SYNTACTIC DEPENDENCIES on top of surface dependencies.
The main characteristics of the deep syntactic annotation scheme are:

* explicitation of subjects of non finite verbs and of adjectives
* neutralization of diathesis alternations
* distribution of dependents over coordinated governors

This led to a first release of the Deep-sequoia corpus (version **1.0**)
(Candito et al., 2014; Perrier et al., 2014).
Annotating the corpus for deep syntax has sometimes led to correct some surface dependencies.

A further step of systematic search for inconsistencies was carried out,
using the Grew system (http://grew.fr).
This led to the release **7.0**.
(Note: the release number (**7.0**) was chosen to get same version numbers for the surface and the deep syntactic annotations of the corpus)

The deep sequoia corpus and the surface sequoia corpus contain the same 3,099 sentences, but note that the original surface corpus (versions prior to 6.0) contained 101 more sentences, that turned out to be duplicates and were thus
subsequently removed (from the EMEA-test part of the corpus).
See the appendix for the ids of the removed sentences.

**MWE and named entities annotation:** Whereas the initial treebank contained grammatical multi-word expressions (MWE) only, the corpus was further annotated for verbal multi-word expressions within the COST project PARSEME, and further for all types of MWE and named entities within the [ANR project PARSEME-FR](http://parsemefr.lif.univ-mrs.fr/doku.php). Note this has led to some modifications of the syntactic representation, cf. the [interaction between syntactic annotation and MWE status](https://gitlab.lis-lab.fr/PARSEME-FR/PARSEME-FR-public/wikis/Interaction-between-syntactic-annotation-and-MWE/Interaction-between-syntactic-annotation-and-MWE-status).

**Supersense annotation:** addition of the coarse semantic class of nouns (including nominal MWEs), within the [FRSEMCOR](https://frsemcor.github.io/FrSemCor/) project, cf. Barque et al. LREC 2020

## References

### Initial version (constituency trees + surface dependencies)
 * **Marie Candito** and **Djamé Seddah**. (2012) [*Le corpus Sequoia : annotation syntaxique et exploitation pour l’adaptation d’analyseur par pont lexical*](https://hal.inria.fr/hal-00698938/document), Proceedings of TALN'2012, Grenoble, France.

 * **Marie Candito** and **Djamé Seddah**. (2012) [*Effectively long-distance dependencies in French: annotation and parsing evaluation*](https://hal.inria.fr/hal-00769625/document), Proceedings of TLT'11, 2012, Lisbon, Portugal.

### Deep syntactic annotations
 * **Marie Candito**, **Guy Perrier**, **Bruno Guillaume**, **Corentin Ribeyre**, **Karën Fort**, **Djamé Seddah** and **Éric de la Clergerie**. (2014) [*Deep Syntax Annotation of the Sequoia French Treebank.*](https://hal.inria.fr/hal-00969191v2/document) Proc. of LREC 2014, Reykjavic, Iceland.

 * **Guy Perrier**, **Marie Candito**, **Bruno Guillaume**, **Corentin Ribeyre**, **Karën Fort** and **Djamé Seddah**. (2014) [*Un schéma d’annotation en dépendances syntaxiques profondes pour le français.*](https://hal.inria.fr/hal-01054407/document) Proc. of TALN 2014, Marseille, France.

### MWE and named entities annotation:

 * **Marie Candito**, **Mathieu Constant**, **Carlos Ramisch**, **Agata Savary**, **Yannick Parmentier**, **Caroline Pasquer** and **Jean-Yves Antoine**. (2017) [*Annotation d'expressions polylexicales verbales en français*](https://hal.archives-ouvertes.fr/hal-01537880/document), Proc. of TALN 2017 - short papers, Orléans

 * **In preparation**: A French corpus annotated for multi-word expressions and named entities.

### Supersense annotation

 * **Lucie Barque**, **Pauline Haas**, **Richard Huyghe**, **Delphine Tribout**, **Marie Candito**, **Benoît Crabbé** and **Vincent Segonne**. (2020) [*Annotating a French Corpus with Supersenses*](https://www.aclweb.org/anthology/2020.lrec-1.724/), Proc. of LREC 2020, Marseille, France.

## Content
The corpus contains 3,099 sentences.

### Number of sentences for each sub-domain:
 * 561 sentences	Europarl	 file= `Europar.550+fct.mrg`
 * 529 sentences	EstRepublicain   file= `annodis.er+fct.mrg`
 * 996 sentences	French Wikipedia file= `frwiki_50.1000+fct.mrg`
 * 574 sentences	EMEA (dev)  	 file= `emea-fr-dev+fct.mrg`
 * 544 sentences	EMEA (test) 	 file= `emea-fr-test+fct.mrg`, among which 101 were removed (because duplicates) in surface version **6.0** and **1.0** deep version.

### Tokenization, multi-word expressions and named entities

* before version **8.0**: the corpus contained grammatical MWEs only, each treated as one token (components separated with an underscore, as in *parce_que*)
* versions **8.xxx**: each grammatical MWE was then represented as separated tokens (with all non-first components attached to the first component with a `dep_cpd` arc.
* from version **9.0**: the MWE and named entities annotated within the PARSEME-FR project were integrated to the corpus, in a separate layer (11th column of CUPT files). MWEs were classified into syntactically regular versus irregular MWEs. Only irregular MWEs have a flat representation with `dep_cpd` arcs. The syntactic representation for named entities and regular MWEs uses regular syntactic dependencies (no `dep_cpd`).


## Dependency formats
For a global view of different dependency formats, see [here](https://deep-sequoia.inria.fr/process/).

### CoNLL format for surface syntax: `sequoia.surf.conll`

```
# sent_id = annodis.er_00449
# text = Une prochaine réunion est prévue mardi
1	Une	un	D	DET	g=f|n=s|s=ind	3	det	_	_
2	prochaine	prochain	A	ADJ	g=f|n=s|s=qual	3	mod	_	_
3	réunion	réunion	N	NC	g=f|n=s|s=c	5	suj	_	_
4	est	être	V	V	m=ind|n=s|p=3|t=pst	5	aux.pass	_	_
5	prévue	prévoir	V	VPP	g=f|m=part|n=s|t=past	0	root	_	_
6	mardi	mardi	N	NC	g=m|n=s|s=c	5	mod	_	_
```

### Extended CoNLL format for both surface and deep syntax: `sequoia.deep_and_surf.conll`

The CoNLL format still contains one token per line, but a given token may have several governors: this is represented by column 7 containing several labels, and column 8 containing several governor ids.
The number of labels in column 7 is the same as the number of governors in column 8 and they are interpreted in parallel:
first label in column 7 corresponds to first governor in column 8, etc.

 * relations subject to diathesis alternations contain a double label (ex: `suj:obj` stands for final subject and canonical object)
 * relations that are only in the deep representation are prefixed by `D:` (ex: `D:suj:suj`)
 * relations that are only in the surf representation are prefixed by `S:` (ex: `S:aux.pass`)
 * a special feature `void=y` is used for tokens removed in the deep syntax (ex: token 4 below)

For instance, the noun *réunion* in the example below is in one hand, both the final subject and canonical object of token 5 (prévue), and in the other hand, the deep subject of token 2 (prochaine).

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

The surf format is obtained by removing relations prefixed by `D:` and by taking the first part of double labels.

A deep format (not distributed here) can be obtained by removing relations prefixed by `S:` and by taking the second part of double labels (see previous releases).

### format with MWE adn Supersense annotations: `sequoia.deep_and_surf.parseme.frsemcor` and `sequoia.surf.parseme.frsemcor`

These files are the same as the previous ones with two more columns:

 * a 11th column for encoding MWE: see the [PARSEME-FR format description](https://gitlab.lis-lab.fr/PARSEME-FR/PARSEME-FR-public/wikis/Corpus-format-description) for more info about MWE and named entities encoding.
 * a 12th column for encoding Supersense: see the [FrSemCor foramt](https://github.com/FrSemCor/FrSemCor/blob/master/fr_semcor_format).

## Appendix
Data split (TALN 2012 experiments)

The "neutral" domain is made of EstRepublicain + Europarl + FrWiki,
and the split into dev and test sets is the following:

```
head -265 annodis.er+fct.mrg >> sequoia-neutre-dev+fct.mrg
head -280 Europar.550+fct.mrg >> sequoia-neutre-dev+fct.mrg
head -498 frwiki_50.1000+fct.mrg >> sequoia-neutre-dev+fct.mrg

tail -264 annodis.er+fct.mrg >> sequoia-neutre-test+fct.mrg
tail -281 Europar.550+fct.mrg >> sequoia-neutre-test+fct.mrg
tail -498 frwiki_50.1000+fct.mrg >> sequoia-neutre-test+fct.mrg
```

Duplicate sentences removed in version 6.0

```
< emea-fr-test_00301
< emea-fr-test_00302
< emea-fr-test_00303
< emea-fr-test_00304
< emea-fr-test_00305
< emea-fr-test_00306
< emea-fr-test_00307
< emea-fr-test_00308
< emea-fr-test_00309
< emea-fr-test_00310
< emea-fr-test_00311
< emea-fr-test_00312
< emea-fr-test_00313
< emea-fr-test_00314
< emea-fr-test_00315
< emea-fr-test_00316
< emea-fr-test_00317
< emea-fr-test_00318
< emea-fr-test_00319
< emea-fr-test_00320
< emea-fr-test_00321
< emea-fr-test_00322
< emea-fr-test_00323
< emea-fr-test_00324
< emea-fr-test_00325
< emea-fr-test_00326
< emea-fr-test_00327
< emea-fr-test_00328
< emea-fr-test_00329
< emea-fr-test_00330
< emea-fr-test_00331
< emea-fr-test_00332
< emea-fr-test_00333
< emea-fr-test_00334
< emea-fr-test_00335
< emea-fr-test_00336
< emea-fr-test_00337
< emea-fr-test_00338
< emea-fr-test_00339
< emea-fr-test_00340
< emea-fr-test_00341
< emea-fr-test_00342
< emea-fr-test_00343
< emea-fr-test_00344
< emea-fr-test_00345
< emea-fr-test_00346
< emea-fr-test_00347
< emea-fr-test_00348
< emea-fr-test_00349
< emea-fr-test_00350
< emea-fr-test_00351
< emea-fr-test_00352
< emea-fr-test_00353
< emea-fr-test_00354
< emea-fr-test_00355
< emea-fr-test_00356
< emea-fr-test_00357
< emea-fr-test_00358
< emea-fr-test_00359
< emea-fr-test_00360
< emea-fr-test_00361
< emea-fr-test_00362
< emea-fr-test_00363
< emea-fr-test_00364
< emea-fr-test_00365
< emea-fr-test_00366
< emea-fr-test_00367
< emea-fr-test_00368
< emea-fr-test_00369
< emea-fr-test_00370
< emea-fr-test_00371
< emea-fr-test_00372
< emea-fr-test_00373
< emea-fr-test_00374
< emea-fr-test_00375
< emea-fr-test_00376
< emea-fr-test_00377
< emea-fr-test_00378
< emea-fr-test_00379
< emea-fr-test_00380
< emea-fr-test_00381
< emea-fr-test_00382
< emea-fr-test_00383
< emea-fr-test_00384
< emea-fr-test_00385
< emea-fr-test_00386
< emea-fr-test_00387
< emea-fr-test_00388
< emea-fr-test_00389
< emea-fr-test_00390
< emea-fr-test_00391
< emea-fr-test_00392
< emea-fr-test_00393
< emea-fr-test_00394
< emea-fr-test_00395
< emea-fr-test_00396
< emea-fr-test_00397
< emea-fr-test_00398
< emea-fr-test_00399
< emea-fr-test_00400
```

