package com.smhrd.web.insect;

import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;

@Data
@AllArgsConstructor
@NoArgsConstructor
public class InsectSummaryResponseDTO {
    private String status;
    private String insect;
    private String solution_summary;
}
