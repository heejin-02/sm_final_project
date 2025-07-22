package com.smhrd.web.admin;

import java.util.List;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Controller;
import org.springframework.ui.Model;
import org.springframework.web.bind.annotation.GetMapping;

@Controller
public class AdminController {

    @Autowired
    private AdminService adminService;
    
    @GetMapping("/admin")
    public String farmList(Model model) {
        List<AdminDTO> farmList = adminService.getFarmList();

        for (AdminDTO farm : farmList) {
            System.out.println(farm.toString());
        }

        model.addAttribute("farmList", farmList);
        return "adminPage";
    }
    
	@GetMapping("/insert")
	public String insert() {
		return "insert";
	}

}
