 fixed_sequence = []
            previous_tag = "O"

            for tag in pred_labels:
                # If there is an I_tag that is not preceded by the correct B_tag, replace it with a B_cat tag
                if tag.startswith('I_') and not previous_tag.endswith(tag[2:]):
                    fixed_sequence.append("B_" + tag[2:])
                else:
                    fixed_sequence.append(tag)
                            
                if tag != "O":
                    previous_tag = tag

            all_predictions.append(fixed_sequence)
            all_ids.append(sample_id)