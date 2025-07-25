package com.smhrd.web.userDetailDTO;

import java.util.List;

import lombok.Data;

@Data
public class FarmDetailResponseDTO {
    private FarmDetailDTO farm;
    private List<GreenhouseDTO> greenhouses;
    private List<CCTVDTO> cctvs;
    private List<ReportDTO> reports;
    private List<FeedbackDTO> feedbacks;
    private List<FarmDetailDTO> classifications;

}
