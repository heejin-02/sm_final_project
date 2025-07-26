package com.smhrd.web.insect;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;

import com.smhrd.web.QcClassification.QcClassificationMapper;

@Service
public class SolutionService {

    @Autowired
    private InsectSolutionService insectSolutionService;

    @Autowired
    private QcClassificationMapper mapper;

    public InsectSummaryResponseDTO getBugReportByImgIdx(Long imgIdx) {
        String insectName = mapper.findInsectNameByImgIdx(imgIdx);
        
        System.out.println("🐛 [DEBUG] imgIdx " + imgIdx + " → 벌레이름: " + insectName);
        String summary = insectSolutionService.getSolutionSummary(insectName);

        return new InsectSummaryResponseDTO("success", insectName, summary);
    }
}
