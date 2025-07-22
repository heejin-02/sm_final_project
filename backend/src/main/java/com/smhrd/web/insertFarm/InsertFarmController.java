package com.smhrd.web.insertFarm;

import org.springframework.stereotype.Controller;
import org.springframework.web.bind.annotation.GetMapping;

@Controller 
public class InsertFarmController {

    @GetMapping("/insertFarm")
    public String insert() {
        return "insertFarm";
    }
    
    @Controller
    public class UserController {

        @GetMapping("/insertUser")
        public String insertUserForm() {
            return "insertUser";
        }
    }

}
