
all:

EXTRA_CXXFLAGS = -Wno-sign-compare
include ../kaldi.mk

BINFILES = vts-model-decode vts-compute-mono-post vts-init-noise dbnvts-init-model \
		   dbnvts-gmm-info \
		   dbnvts-est-gmm vts-compute-likes vts-est-noise vts-align-compiled vts-apply-global-cmvn \
		   copy-feats-to-matlab compute-utt-noise vts-noise-decode compute-average-noise \
		   vtsbnd-acc-stats vtsbnd-forward dbnvts2-acc-stats dbnvts2-neg-gauss dbnvts2-forward \
		   dbnvts2-test-kl dbnvts2-test-gmm dbnvts2-sum-accs dbnvts2-sum-priors kl-divergence-gauss \
		   dbnvts2-acc-stats-act dbnvts2-compensate-model dbnvts2-joint-forward-clean \
		   dbnvts2-joint-forward dbnvts2-compute-llr dbnvts2-forward-interpolate remove-trans-model \
		   vts-acc-gmm-stats-ali vts-gmm-sum-accs vts-gmm-est vts-compute-obj vts-sum-obj vts-apply-global-cmvn-fbank \
		   vts-feats vts-feats-fbank noise-mfc2fbk feats-append-noise gmm-global-get-frame-comp-scores \
		   vts-init-global-noise vts-est-global-noise vts-global-noise-decode
 
OBJFILES =

all:  $(BINFILES)


TESTFILES =


$(BINFILES): ../vts/kaldi-vts.a ../nnet/kaldi-nnet.a ../cudamatrix/cuda-matrix.a ../lat/kaldi-lat.a ../decoder/kaldi-decoder.a  ../feat/kaldi-feature.a ../transform/kaldi-transform.a ../gmm/kaldi-gmm.a ../hmm/kaldi-hmm.a ../tree/kaldi-tree.a ../matrix/kaldi-matrix.a  ../util/kaldi-util.a ../base/kaldi-base.a 



# Rule below would expand to, e.g.:
# ../base/kaldi-base.a:
# 	make -c ../base kaldi-base.a
# -c option to make is same as changing directory.
%.a:
	$(MAKE) -C ${@D} ${@F}

clean:
	-rm -f *.o *.a $(TESTFILES) $(BINFILES)

test: $(TESTFILES)
	for x in $(TESTFILES); do ./$$x >/dev/null 2>&1 || { echo "***test $$x failed***"; exit 1; }; done  
	echo Tests succeeded

.valgrind: $(TESTFILES)


depend:
	-$(CXX) -M $(CXXFLAGS) *.cc > .depend.mk  

# removing automatic making of "depend" as it's quite slow.
#.depend.mk: depend

-include .depend.mk

