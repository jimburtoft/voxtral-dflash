| Set | Role | Dataset / config | Split | Clips | Hours | Speakers | Pool rows | Dropped by filters (select stage) | Decode-stage rejects | Eligible after filters |
|---|---|---|---|---|---|---|---|---|---|---|
| ls_train | train | `openslr/librispeech_asr` / all | train.clean.100 | 2,720 | 9.60 | 234 | 281,241 | chapter_reserved_for_other_corpus=131059, task001_flagged_list=36 | none | 150,146 |
| ls_train | train | `openslr/librispeech_asr` / all | train.clean.360 | 9,843 | 34.52 | 849 | 281,241 | chapter_reserved_for_other_corpus=131059, task001_flagged_list=36 | none | 150,146 |
| ls_train | train | `openslr/librispeech_asr` / all | train.other.500 | 12,437 | 41.78 | 1071 | 281,241 | chapter_reserved_for_other_corpus=131059, task001_flagged_list=36 | none | 150,146 |
| ltts_train | train | `mythicinfinity/libritts_r` / all | train.clean.100 | 1,614 | 2.81 | 219 | 354,693 | chapter_reserved_for_other_corpus=186144, task001_flagged_list=43 | decoded_duration_out_of_range=747 | 168,506 |
| ltts_train | train | `mythicinfinity/libritts_r` / all | train.clean.360 | 5,710 | 10.65 | 771 | 354,693 | chapter_reserved_for_other_corpus=186144, task001_flagged_list=43 | decoded_duration_out_of_range=747 | 168,506 |
| ltts_train | train | `mythicinfinity/libritts_r` / all | train.other.500 | 7,676 | 13.70 | 991 | 354,693 | chapter_reserved_for_other_corpus=186144, task001_flagged_list=43 | decoded_duration_out_of_range=747 | 168,506 |
| mls_train | train | `parler-tts/mls_eng` / default | train | 20,000 | 82.69 | 5193 | 10,808,037 | text_overlap_vs_eval=15 | none | 199,985 |
| cv_train | train | `fsicoli/common_voice_17_0` / en | train | 40,000 | 65.44 | 37475 | 1,101,170 | meta_duration_outside_1_30s=4, task001_flagged_list=83 | none | 1,101,083 |
| vp_train | train | `facebook/voxpopuli` / en | train | 30,000 | 82.67 | 320 | 182,482 | speaker_in_eval_droplist=47276, speaker_unknown_None=64260, task001_flagged_list=4 | decoded_duration_out_of_range=357 | 70,942 |
| ps_train | train | `MLCommons/peoples_speech` / clean | train | 25,000 | 95.93 | - | 1,501,271 | source_license_not_allowed=592, text_overlap_vs_eval=1 | none | 1,500,678 |
| ami_train | train | `edinburghcstr/ami` / ihm | train | 20,000 | 24.20 | 152 | 108,502 | meeting_IB4005_dropped=421, meta_duration_outside_1.5_30s=51652, task001_flagged_list=7 | none | 56,422 |
| ls_eval | eval | `openslr/librispeech_asr` / all | test.clean | 2,620 | 5.40 | 40 | 5,559 | none | none | 5,559 |
| ls_eval | eval | `openslr/librispeech_asr` / all | test.other | 2,939 | 5.34 | 33 | 5,559 | none | none | 5,559 |
| cv_eval | eval | `fsicoli/common_voice_17_0` / en | test | 2,000 | 3.27 | 1939 | 16,393 | none | none | 16,393 |
| vp_eval | eval | `facebook/voxpopuli` / en | test | 1,842 | 4.93 | 285 | 1,842 | none | none | 1,842 |
| fleurs_eval | eval | `google/fleurs` / en_us | test | 647 | 1.77 | - | 647 | none | none | 647 |
| ami_eval | eval | `edinburghcstr/ami` / ihm | test | 2,000 | 2.44 | 16 | 12,643 | meta_duration_outside_1.5_30s=6304 | none | 6,339 |
| ami_longform | eval | `edinburghcstr/ami` ihm segments + official Mix-Headset audio | test | 60 | 2.95 | 16 meetings | - | - | - | - |
| **total train** | | | | **175,000** | **464.0** | | | | | |
| **total eval** | | | | **12,108** | **26.1** | | | | | |
