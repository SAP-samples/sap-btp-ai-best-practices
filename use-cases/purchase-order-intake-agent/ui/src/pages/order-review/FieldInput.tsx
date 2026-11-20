/**
 * Labelled input for one order field with confidence %, value state and explanation tooltip.
 */
import { Input, Label, SuggestionItem } from "@ui5/webcomponents-react";

import type { FieldState } from "../../services/orders";
import { messageOf, tooltipOf, valueStateOf } from "./fieldState";

interface Props {
  /** Field label. */
  label: string;
  /** Current (possibly edited) value. */
  value: string | number | null | undefined;
  /** Validation / extraction metadata of the field. */
  state?: FieldState;
  /** Called with the new text value. */
  onChange?: (value: string) => void;
  /** Disable editing. */
  readOnly?: boolean;
  /** Input type (text, number, date). */
  type?: "Text" | "Number";
  /** Optional suggestions (value help); text = value, additionalText = description. */
  suggestions?: { text: string; additionalText?: string }[];
  /** Called on every keystroke (to fetch suggestions). */
  onTyping?: (value: string) => void;
  /** Grid column span class. */
  wide?: boolean;
}

/**
 * Renders the field.
 *
 * @param props - See `Props`.
 * @returns Label + input block.
 */
export default function FieldInput({ label, value, state, onChange, readOnly, type = "Text", suggestions, onTyping, wide }: Props) {
  const confidence = state?.confidence;
  const lowConfidence = confidence !== null && confidence !== undefined && confidence < 0.8;
  const message = messageOf(state);
  const valueState = valueStateOf(state);
  return (
    <div className={`review-field${wide ? " review-field--wide" : ""}`} title={tooltipOf(state)}>
      <div className="review-field-label">
        <Label showColon>{label}</Label>
        {confidence !== null && confidence !== undefined && (
          <span className={`review-confidence${lowConfidence ? " review-confidence--low" : ""}`}>
            {Math.round(confidence * 100)}%
          </span>
        )}
      </div>
      <Input
        value={value === null || value === undefined ? "" : String(value)}
        type={type}
        readonly={readOnly}
        valueState={valueState}
        valueStateMessage={message ? <div>{message}</div> : undefined}
        showSuggestions={Boolean(suggestions)}
        onInput={(event) => {
          onTyping?.(event.target.value);
          onChange?.(event.target.value);
        }}
        onChange={(event) => onChange?.(event.target.value)}
      >
        {suggestions?.map((s) => <SuggestionItem key={s.text} text={s.text} additionalText={s.additionalText} />)}
      </Input>
    </div>
  );
}
