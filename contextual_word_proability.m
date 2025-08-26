%% For each word, compute model outputs and append
for i=1:nWordsChapter
    % Display progress after every 100 words
    if mod(i,100)==0
        fprintf('\nProcessing file %d (%d out of %d items)...\n\n',f,i,nWordsChapter);
        pause(.1);
    end

    % If not silence or noise
    if stim.phon{i}~="<p:>" && stim.phon{i}~="<usb>"
        % Get phonology for current word (from TextGrid)
        phonCurrent = stim.phon{i};

        % Get current context
        contextCurrent = stim.orth(1:i-1); % Get all words up to penultimate word
        contextCurrent(strcmp(contextCurrent,'')) = []; % ignore pauses
        contextCurrent = strjoin(contextCurrent,' '); % convert from cell array of words to string

        % Get next word probabilities using GPT2
        if isempty(contextCurrent)
            lexicon.context = ones(nWords,1)*(1/nWords);
        else
            ids = mdl.Tokenizer.encode(contextCurrent);
            nTokensExceed = numel(ids)-1024;
            if nTokensExceed>0; ids = ids(1+nTokensExceed:end); end
            logits = gpt2.model(ids,pasts,mdl.Parameters);
            logits = logits(:,end);
            logits = double(extractdata(logits));
            probsGPT = exp(logits)./sum(exp(logits),1);

            % ---- Minimal change: top-p includes the first token that pushes cumprob >= 0.95 ----
            p_threshold = 0.95;
            [sorted_probs, sorted_indices] = sort(probsGPT, 'descend');
            cumulative_probs = cumsum(sorted_probs);
            cutoff_idx = find(cumulative_probs >= p_threshold, 1, 'first');
            if isempty(cutoff_idx)
                cutoff_idx = numel(sorted_indices);
            end
            ind2compute = sorted_indices(1:cutoff_idx);
            % -------------------------------------------------------------------------------------

            lexicon.context = zeros(nWords, 1);

            for ii = 1:numel(ind2compute)
                gpt_lex = lower(strip(mdl.Tokenizer.decode(ind2compute(ii))));
                logicalMatch = strcmp(list_lex, gpt_lex);
                if ~any(logicalMatch)
                    continue;
                end
                lexicon.context(logicalMatch) = lexicon.context(logicalMatch) + probsGPT(ind2compute(ii))/sum(logicalMatch);
            end

            tailMask = lexicon.context == 0;
            nTail = sum(tailMask);
            tailMass = 1 - sum(lexicon.context);
            if nTail > 0
                lexicon.context(tailMask) = tailMass / nTail;
            end

            if abs(sum(lexicon.context)-1) > 1e-6
                error('Context probabilities do not sum to 1!')
            end
        end

        % Run model for current word (GPT priors)
        [wordProbs,segmentProbs] = es_prediction_sim(phonCurrent,lexicon.segments,lexicon.phon,lexicon.context,segmentConfusionMatrix,1,1,assumeIdeal);
        [wordProbs,segmentProbs] = es_prediction_information(phonCurrent,lexicon.segments,lexicon.phon,wordProbs,segmentProbs);
        if isinf(wordProbs.surprisal); error('Found Inf in suprisal values!'); end

        model_context.segments.input(stim.segmentOnsets{i},:) = segmentProbs.l';
        model_context.segments.pred(stim.segmentOnsets{i},:) = segmentProbs.p';
        model_context.segments.ss(stim.segmentOnsets{i},:) = segmentProbs.ps';
        model_context.segments.pe(stim.segmentOnsets{i},:) = segmentProbs.pe';
        model_context.segments.peSumAbs(stim.segmentOnsets{i},1) = sum(abs(segmentProbs.pe),1)';
        model_context.segments.surprisal(stim.segmentOnsets{i},1) = segmentProbs.surprisal';
        model_context.segments.entropy(stim.segmentOnsets{i},1) = segmentProbs.entropy';
        model_context.segments.kld(stim.segmentOnsets{i},1) = segmentProbs.kld';

        model_context.words.surprisal(stim.wordOnsets(i),1) = wordProbs.surprisal';
        model_context.words.entropy(stim.wordOnsets(i),1) = wordProbs.entropy';

        % Run model for current word (word frequency priors)
        [wordProbs,segmentProbs] = es_prediction_sim(phonCurrent,lexicon.segments,lexicon.phon,lexicon.freq,segmentConfusionMatrix,1,1,assumeIdeal);
        [wordProbs,segmentProbs] = es_prediction_information(phonCurrent,lexicon.segments,lexicon.phon,wordProbs,segmentProbs);
        if isinf(wordProbs.surprisal); error('Found Inf in suprisal values!'); end

        model_freq.segments.input(stim.segmentOnsets{i},:) = segmentProbs.l';
        model_freq.segments.pred(stim.segmentOnsets{i},:) = segmentProbs.p';
        model_freq.segments.ss(stim.segmentOnsets{i},:) = segmentProbs.ps';
        model_freq.segments.pe(stim.segmentOnsets{i},:) = segmentProbs.pe';
        model_freq.segments.peSumAbs(stim.segmentOnsets{i},1) = sum(abs(segmentProbs.pe),1)';
        model_freq.segments.surprisal(stim.segmentOnsets{i},1) = segmentProbs.surprisal';
        model_freq.segments.entropy(stim.segmentOnsets{i},1) = segmentProbs.entropy';
        model_freq.segments.kld(stim.segmentOnsets{i},1) = segmentProbs.kld';

        model_freq.words.surprisal(stim.wordOnsets(i),1) = wordProbs.surprisal';
        model_freq.words.entropy(stim.wordOnsets(i),1) = wordProbs.entropy';
    end
end
