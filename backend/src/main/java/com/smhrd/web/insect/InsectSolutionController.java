package com.smhrd.web.insect;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

@RestController
@RequestMapping("/api/insect")
public class InsectSolutionController {

    @Autowired
    private SolutionService solutionService;

    @GetMapping("/solution")
    public ResponseEntity<InsectSummaryResponseDTO> getBugSolution(@RequestParam Long imgIdx) {
        InsectSummaryResponseDTO response = solutionService.getBugReportByImgIdx(imgIdx);
        
        return ResponseEntity.ok(response);
    }
}
