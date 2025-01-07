import os
import torch
import logging
from typing import Optional, List, Union
from transformers import (
    LlamaForCausalLM,
    DynamicCache,
    PreTrainedTokenizer,
)
from . import CoconutConfig
from ..dataset import split_sequences

logger = logging.getLogger(__name__)


class CoconutLlamaForCausalLM(LlamaForCausalLM):

    def __init__(self, config):
        super().__init__(config)
        self.tokenizer: PreTrainedTokenizer = None
        self.coconut_config: CoconutConfig = CoconutConfig.from_dict(
            config.coconut_config
        )
        self.current_stage = 0
        self.debug = os.environ.get("DEBUG") == "1"

    def thoughts_forward(
            self,
            num_thoughts : int = 1,
            inputs_embeds : Optional[torch.Tensor] = None,
            attention_mask : Optional[torch.Tensor] = None,
            past_key_values : Optional[List[torch.Tensor]] = None,
    ) -> List:
        all_thought_outputs = []

        for t in range(num_thoughts):
            outputs = self.model.forward(
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                use_cache=True,
                return_dict= True,
            )

            # The inputs for the next thought will be the current hidden state
            inputs_embeds = outputs.last_hidden_state[:, -1:, :]
            attention_mask = torch.cat(
                (
                    attention_mask,
                    torch.ones(
                        (inputs_embeds.shape[0], 1),
                        dtype=attention_mask.dtype,
                        device=attention_mask.device,
                    ),
                ),
                dim=1,
            )
            past_key_values = outputs.past_key_values

            if self.debug:
                all_thought_outputs.append(
                    self.hidden_states_to_token(inputs_embeds, lm_head=True)
                )
        
        return all_thought_outputs
    
    def infer_forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[DynamicCache, List[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        num_logits_to_keep: int = 0,
        **loss_kwargs,
    ):
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )
        batch_size = input_ids.shape[0]

        if input_ids.shape[1] > 1:
            input_ids = torch.concat(
                [
                    input_ids,
                    torch.tensor(
                        [[self.coconut_config.bot_id]] * batch_size,
                        device=input_ids.device,
                    ),
                ],
                dim=1,
            )
            attention_mask = torch.concat(
                [
                    attention_mask,
                    torch.ones(
                        attention_mask.shape[0], 1, device=attention_mask.device
                    ),
                ],
                dim=1,
            )
        
        if past_key_values is None:
            past_key_values = DynamicCache()

        if self.coconut_config.stages - 1 > 0 and input_ids.shape[1] > 1:
            
            num_thoughts = (
                (self.coconut_config.stages-1) * self.coconut_config.continuous_thoughts
            )

            inputs_embeds = self.get_input_embeddings()(input_ids)

            all_thought_outputs = self.thoughts_forward(
                num_thoughts = num_thoughts,
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
            )

            inputs_embeds = self.get_input_embeddings()(
                torch.tensor(
                    [[self.coconut_config.eot_id]] * batch_size,
                    device=inputs_embeds.device
                )
            )

            additional_mask = torch.ones(
                (batch_size, num_thoughts - 1),
                dtype=attention_mask.dtype,
                device=attention_mask.device,
            )

            # 기존 attention_mask와 추가된 mask를 concatenate
            attention_mask = torch.cat((attention_mask, additional_mask), dim=1)

            outputs = super().forward(
                input_ids=None,
                attention_mask=attention_mask,
                position_ids=None,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                labels=labels,
                use_cache=True,
                output_attentions=output_attentions,
                output_hidden_states=True,
                return_dict=True,
                num_logits_to_keep=num_logits_to_keep,
            )

            if self.debug:
                self._print_thought_and_final_tokens(
                    outputs.logits, all_thought_outputs
                )

        else:
            # Standard forward pass
            outputs = super().forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=None,
                labels=labels,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=True,
                cache_position=cache_position,
                num_logits_to_keep=num_logits_to_keep,
                **loss_kwargs,
            )

        return outputs
    
    def train_forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[DynamicCache, List[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        num_logits_to_keep: int = 0,
        **loss_kwargs,
    ):
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        # Split sequences into thought and language parts
        (
            thought_ids,
            language_ids,
            thought_mask,
            _,
            _,
            language_labels,
        ) = split_sequences(input_ids, attention_mask, labels, self.coconut_config)

        all_thought_outputs = []

        if past_key_values is None:
            past_key_values = DynamicCache()

        if self.current_stage > 0:
            num_thoughts = self.current_stage * self.coconut_config.continuous_thoughts
            inputs_embeds = self.get_input_embeddings()(thought_ids)

            all_thought_outputs = self.thoughts_forward(
                num_thoughts, inputs_embeds, thought_mask, past_key_values
            )

            inputs_embeds = self.get_input_embeddings()(language_ids)

            # we fix the mask and labels lengths by inserting between <bot><eot>
            insert_indices = (input_ids == self.coconut_config.eot_id).nonzero(
                as_tuple=True
            )[1]

            attention_mask, labels = self._insert_thoughts_with_gather(
            attention_mask = attention_mask, 
            labels = labels, 
            insert_indices = insert_indices, 
            num_thoughts = num_thoughts,
            )

            outputs = super().forward(
                input_ids=None,
                attention_mask=attention_mask,
                position_ids=None,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                labels=language_labels,
                use_cache=True,
                output_attentions=output_attentions,
                output_hidden_states=True,
                return_dict=True,
                cache_position=cache_position,
                num_logits_to_keep=num_logits_to_keep,
            )

            if self.debug:
                tokens = []
                for i, (id, mask, label) in enumerate(
                    zip(
                        input_ids[0].tolist(),
                        attention_mask[0].tolist(),
                        labels[0].tolist(),
                    )
                ):
                    tokens.append(f"<{self.tokenizer.decode(id)}> ({mask}, {label})")
                    if i == insert_idx:
                        tokens.append(f"<[LATENT THOUGHT]> ({mask}, {label})")
                print(" ".join(tokens))
                self._print_thought_and_final_tokens(
                    outputs.logits, all_thought_outputs
                )
        else:
            # Standard forward pass
            outputs = super().forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=None,
                labels=labels,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=True,
                cache_position=cache_position,
                num_logits_to_keep=num_logits_to_keep,
                **loss_kwargs,
            )

        return outputs



    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[DynamicCache, List[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        num_logits_to_keep: int = 0,
        **loss_kwargs,
    ):
        if self.training:
            forward_fn = self.train_forward
        else:
            forward_fn = self.infer_forward

        outputs = forward_fn(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
            num_logits_to_keep=num_logits_to_keep,
            **loss_kwargs,
        )

        """
        transforemrs.models.llama.modeling_llama
        class LlamForCausalLM
        def forward

        hidden_states = outputs[0]
        # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
        logits = self.lm_head(hidden_states[:, -num_logits_to_keep:, :])

        loss = None
        if labels is not None:
            loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.vocab_size, **kwargs)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
        """

        return outputs    
   
    @torch.no_grad()
    def hidden_states_to_token(
        self,
        logits: torch.Tensor,
        lm_head: bool = False,
    ) -> List:
        if lm_head:
            logits = self.lm_head(logits)
        probs = torch.nn.functional.softmax(logits[:,-1,:], dim=-1)
        top_probs, top_indices = torch.topk(probs, 3)

        tokens = []

        for prob, token_id in zip(top_probs.squeeze(), top_indices.squeeze()):
            tokens.append(
                {
                    "token": self.tokenizer.decode(token_id.item()),
                    "prob": prob.item(),
                    "token_id": token_id.item(),
                }
            )
        
        return tokens
    
    def _print_thought_and_final_tokens(
        self, logits: torch.Tensor, all_thought_outputs: List[torch.Tensor]
    ):
        final_thoughts = []
        final_token = self.hidden_states_to_token(logits)[0]
        for i, sampled_tokens in enumerate(all_thought_outputs):
            tokens_formatted = []
            for j, token in enumerate(sampled_tokens):
                tokens_formatted.append(
                    f"t_{i},{j}: [{token['token'].strip()}] (p: {token['prob']:.3f})"
                )
            final_thoughts.append((" || ").join(tokens_formatted))
        print("\n".join(final_thoughts))
        print(
            f"t_final: [{final_token['token'].strip()}] (p: {final_token['prob']:.3f})"
        )

    def _insert_thoughts_with_gather(
        self,
        attention_mask: torch.Tensor,
        labels: torch.Tensor,
        insert_indices: torch.Tensor,
        num_thoughts: int,
    ):
        """
        insert_indices[b] 위치에 (num_thoughts - 1) 만큼의 구간을 삽입해
        attention_mask와 labels 텐서를 확장(gather 기반)하는 예시 코드.

        Parameters
        ----------
        attention_mask : (batch_size, seq_len)
        labels         : (batch_size, seq_len)
        insert_indices : (batch_size,)         각 배치(b)마다 <eot> 위치
        num_thoughts   : int                   (삽입할 구간은 num_thoughts-1 길이)

        Returns
        -------
        new_attention_mask : (batch_size, seq_len + num_thoughts - 1)
        new_labels         : (batch_size, seq_len + num_thoughts - 1)
        """
        device = attention_mask.device
        batch_size, seq_len = attention_mask.shape
        num_insert = num_thoughts - 1  # 삽입 구간 길이

        # 최종 확장된 시퀀스 길이
        new_seq_len = seq_len + num_insert

        # -- 1) index_map: [batch_size, new_seq_len]
        # 각 위치(행: 배치 b, 열: 새 시퀀스 c)에 대해
        # "원본 attention_mask/labels에서 가져올 인덱스"를 저장
        index_map = torch.zeros((batch_size, new_seq_len), dtype=torch.long, device=device)

        # range_new: [batch_size, new_seq_len] -> 각 배치별로 0..(new_seq_len-1)
        range_new = torch.arange(new_seq_len, device=device).unsqueeze(0)
        range_new = range_new.expand(batch_size, new_seq_len)

        # 왼쪽(left) 마스크: range_new < insert_indices
        #   -> '삽입 위치보다 왼쪽은 "원본 인덱스 = 현재 인덱스"'
        left_mask = (range_new < insert_indices.unsqueeze(1))
        # 오른쪽(right) 마스크: range_new >= insert_indices + num_insert
        #   -> '삽입 영역을 지나간 곳은 "원본 인덱스 = 현재 인덱스 - num_insert"'
        right_mask = (range_new >= (insert_indices.unsqueeze(1) + num_insert))
        # 삽입(insertion) 마스크: 나머지 (위 두 마스크의 보수)
        insert_mask = ~(left_mask | right_mask)

        # 왼쪽 영역은 index_map = c (즉, 그대로)
        index_map[left_mask] = range_new[left_mask]
        # 오른쪽 영역은 index_map = c - num_insert
        #   (삽입된 구간만큼 오른쪽으로 밀리므로 원본 인덱스는 c - num_insert)
        index_map[right_mask] = range_new[right_mask] - num_insert
        # 삽입 영역은 일단 dummy로 0 (혹은 0 이하나 seq_len 이상 등)으로 두고,
        # gather 후 별도로 값(1 or -100)을 채울 예정
        index_map[insert_mask] = 0  # dummy index

        # -- 2) gather로 원본에서 가져오기
        gathered_attention_mask = attention_mask.gather(dim=1, index=index_map)
        gathered_labels = labels.gather(dim=1, index=index_map)

        # -- 3) 실제 삽입 구간을 1, -100으로 채우기
        # (삽입 구간: insert_mask == True)
        gathered_attention_mask[insert_mask] = 1
        gathered_labels[insert_mask] = -100

        return gathered_attention_mask, gathered_labels
